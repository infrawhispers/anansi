use std::cmp::max;
use std::fs;
use std::path::PathBuf;

use anyhow::bail;
use ndarray::Dim;
use ndarray::{array, s, Array, Array1};
use ort::{
    tensor::DynOrtTensor, tensor::FromArray, tensor::InputTensor, tensor::OrtOwnedTensor,
    GraphOptimizationLevel, Session, SessionBuilder,
};
use tokenizers::tokenizer::Tokenizer;

use crate::embedder::{Embedder, EmbedderParams, EmebeddingRequest, InstructorParams};
use crate::utils::download_model_sync;

pub struct InstructorEmbedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder for InstructorEmbedder {
    fn new(params: &EmbedderParams) -> anyhow::Result<Self> {
        InstructorEmbedder::new(params)
    }
    fn encode(&self, req: &EmebeddingRequest) -> anyhow::Result<Vec<Vec<f32>>> {
        let req_instructor: &InstructorParams = match req {
            EmebeddingRequest::InstructorRequest { params } => &params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        self.encode(req_instructor.instructions, req_instructor.text)
    }
}

static EMBEDDING_LENGTH: usize = 768;
static S3_BUCKET_URI: &'static str = "https://d1wz516niig2xr.cloudfront.net/models.getanansi.com/";

impl InstructorEmbedder {
    fn gen_encodings(
        &self,
        instruction: &str,
        text: &str,
    ) -> anyhow::Result<(
        Array<i64, Dim<[usize; 2]>>,
        Array<i64, Dim<[usize; 2]>>,
        Array1<i64>,
    )> {
        let context_encoding;
        match self.tokenizer.encode(instruction, true) {
            Ok(res) => context_encoding = res,
            Err(err) => {
                bail!("unable to tokenize the input: {}", err)
            }
        }
        let mut ctx_masks_sum: i64 = 0;
        context_encoding
            .get_attention_mask()
            .iter()
            .for_each(|x| ctx_masks_sum += *x as i64);
        ctx_masks_sum -= 1;
        let concaction = format!("{}{}", instruction, text);
        let text_encoding;
        match self.tokenizer.encode(concaction, true) {
            Ok(res) => text_encoding = res,
            Err(err) => {
                bail!("unable to tokenize the input: {}", err)
            }
        }
        let i_tvals = text_encoding.get_ids();
        let i_atten_mask = text_encoding.get_attention_mask();
        let tvals_nonzero = i_tvals.iter().filter(|&n| *n != 0).count();
        let atten_mask_nonzero = i_atten_mask.iter().filter(|&n| *n != 0).count();
        let mut a = Array::<i64, _>::zeros((1, tvals_nonzero));
        let mut b = Array::<i64, _>::zeros((1, atten_mask_nonzero));
        let c: Array1<i64> = array![ctx_masks_sum];
        // this is safe because our embeddings are of the form {x1, x2, x3...0, 0, 0, 0}
        for i in 0..tvals_nonzero {
            a[[0, i]] = i_tvals[i] as i64
        }
        for i in 0..atten_mask_nonzero {
            b[[0, i]] = i_atten_mask[i] as i64
        }
        Ok((a, b, c))
    }

    fn encode(&self, instructions: &[String], text: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        if instructions.len() != text.len() {
            bail!(
                "text.len != instructions.len - ({} != {})",
                text.len(),
                instructions.len()
            )
        }

        let (mut max_a, mut max_b, mut max_c) = (0, 0, 0);
        let mut all_a: Vec<Array<i64, Dim<[usize; 2]>>> = Vec::new();
        let mut all_b: Vec<Array<i64, Dim<[usize; 2]>>> = Vec::new();
        let mut all_c: Vec<Array1<i64>> = Vec::new();
        for i in 0..instructions.len() {
            let (a, b, c) = self.gen_encodings(&instructions[i], &text[i])?;

            max_a = max(max_a, a.len());
            max_b = max(max_b, b.len());
            max_c = max(max_c, c.len());

            all_a.push(a);
            all_b.push(b);
            all_c.push(c);
        }
        // TODO(infrawhispers) - this is a hack as the current ONNX model does not play nicely
        // with single usage - we may need to rexplore tweaking the export process of the model
        if instructions.len() == 1 {
            let (a, b, c) = self.gen_encodings(&instructions[0], &text[0])?;

            max_a = max(max_a, a.len());
            max_b = max(max_b, b.len());
            max_c = max(max_c, c.len());

            all_a.push(a);
            all_b.push(b);
            all_c.push(c);
        }

        let mut a = Array::<i64, _>::zeros((all_a.len(), max_a));
        let mut b = Array::<i64, _>::zeros((all_b.len(), max_b));
        let mut c = Array1::<i64>::zeros(all_c.len());
        for idx in 0..all_a.len() {
            a.slice_mut(s![idx..idx + 1, 0..all_a[idx].len()])
                .assign(&all_a[idx]);
            b.slice_mut(s![idx..idx + 1, 0..all_b[idx].len()])
                .assign(&all_b[idx]);
            c[idx] = all_c[idx][0];
        }
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self.session.run([
            InputTensor::from_array(a.into_dyn()),
            InputTensor::from_array(b.into_dyn()),
            InputTensor::from_array(c.into_dyn()),
        ])?;

        let embedding: OrtOwnedTensor<f32, _> = outputs[outputs.len() - 1].try_extract().unwrap();
        let embedding = embedding.view().to_owned();
        let mut result: Vec<Vec<f32>> = Vec::with_capacity(instructions.len());
        for idx in 0..instructions.len() {
            result.push(
                embedding
                    .slice(s![idx..idx + 1, 0..EMBEDDING_LENGTH])
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
            );
        }
        Ok(result)
    }
    pub fn new(params: &EmbedderParams) -> anyhow::Result<Self> {
        let model_name = "instructor_w_ctx_mask_v2.onnx";
        if !params.model_path.exists() {
            fs::create_dir_all(params.model_path)?;
        }
        let model_file_path = PathBuf::from(params.model_path).join(model_name);
        if !model_file_path.exists() {
            download_model_sync(
                &format!("{}/instructor/{}", S3_BUCKET_URI, model_name),
                true,
                &model_file_path,
                "8f829d4c1714de7c25872fdafab633aa",
            )?;
            // let runtime = Runtime::new()?;
            // tokio::runtime::Runtime::new()?.block_on(download_model(
            //     &format!("{}/instructor/{}", S3_BUCKET_URI, model_name),
            //     true,
            //     &model_file_path,
            //     "13bd03638da80fc57f83e1b77abd64f8",
            // ))?;
        }
        let session = SessionBuilder::new(&params.ort_environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_inter_threads(params.num_threads)?
            .with_parallel_execution(true)?
            .with_model_from_file(model_file_path)?;

        let tokenizer: Tokenizer;
        match Tokenizer::from_pretrained("hkunlp/instructor-large", None) {
            Ok(tk) => tokenizer = tk,
            Err(err) => {
                bail!("unable to create a tokenizer: {}", err);
            }
        }
        Ok(InstructorEmbedder {
            session: session,
            tokenizer: tokenizer,
        })
    }
}