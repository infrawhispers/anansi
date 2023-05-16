use std::cmp::max;
use std::fs;
use std::path::PathBuf;

use anyhow::bail;
use ndarray::Dim;
use ndarray::{s, Array, ArrayView, Axis};
use ort::tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor};
use ort::{Session, SessionBuilder};
use phf::phf_map;
use tokenizers::tokenizer::Tokenizer;
use tracing::info;

use crate::utils::download_model_sync;

use super::embedder::{E5Params, Embedder, EmbedderParams, EmebeddingRequest};

pub struct E5Embedder {
    session: Session,
    tokenizer: Tokenizer,
}

impl Embedder for E5Embedder {
    fn new(params: &EmbedderParams) -> anyhow::Result<Self> {
        E5Embedder::new(params)
    }
    fn encode(&self, req: &EmebeddingRequest) -> anyhow::Result<Vec<Vec<f32>>> {
        let req: &E5Params = match req {
            EmebeddingRequest::E5Request { params } => &params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        self.encode(req.text)
    }
}

static S3_BUCKET_URI: &'static str = "https://d1wz516niig2xr.cloudfront.net";
#[derive(Clone, Copy)]
struct ModelDetails {
    pub filename: &'static str,
    pub hash_val: &'static str,
    pub tokenizer_name: &'static str,
}
static E5_MODELS: phf::Map<&'static str, ModelDetails> = phf_map! {
    "E5_LARGE" => ModelDetails {
        filename: "e5_large.onnx",
        hash_val: "26d08efc54abf7688bd34361e0d1a021",
        tokenizer_name: "intfloat/e5-large",
    },
    "E5_BASE" => ModelDetails {
        filename: "e5_base.onnx",
        hash_val: "2e3f358ff23c7b3a817d0db31497148f",
        tokenizer_name: "intfloat/e5-base",
    },
    "E5_SMALL" => ModelDetails {
        filename: "e5_small.onnx",
        hash_val: "3d421dc72859a723068c106415cdebf2",
        tokenizer_name: "intfloat/e5-small",
    },
};

impl E5Embedder {
    fn average_pool(
        &self,
        mut last_hidden_state: Array<f32, Dim<ndarray::IxDynImpl>>,
        atten_mask: Array<i64, Dim<[usize; 2]>>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        // invert the attention mask:
        // ~attention_mask[..., None].bool()
        let mut atten_mask_inv = atten_mask.clone();
        let atten_mask_shape_rows = atten_mask.shape()[0];
        let atten_mask_shape_cols = atten_mask.shape()[1];
        for idx in 0..atten_mask_shape_rows {
            let inverted: Vec<i64> = atten_mask_inv
                .slice(s![idx..idx + 1, 0..atten_mask_shape_cols])
                .iter()
                .map(|&s| if s == 0 { 1 } else { 0 })
                .collect::<Vec<i64>>();
            atten_mask_inv
                .slice_mut(s![idx..idx + 1, 0..atten_mask_shape_cols])
                .assign(&ArrayView::from(&inverted));
        }
        // now fill in the hidden_s with 0.0 if the rows are
        // empty!
        let hidden_s_rows = last_hidden_state.shape()[0];
        let hidden_s_dims = last_hidden_state.shape()[2];
        for idx_d in 0..hidden_s_dims {
            for idx_r in 0..hidden_s_rows {
                let filled: Vec<f32> = last_hidden_state
                    .slice(s![idx_r..idx_r + 1, .., idx_d..idx_d + 1])
                    .iter()
                    .enumerate()
                    .map(|(idx_c, &s)| {
                        if atten_mask_inv[[idx_r, idx_c]] == 1 {
                            0.0
                        } else {
                            s
                        }
                    })
                    .collect::<Vec<f32>>();
                last_hidden_state
                    .slice_mut(s![idx_r..idx_r + 1, .., idx_d..idx_d + 1])
                    .assign(&ArrayView::from_shape((1, filled.len(), 1), &filled)?);
            }
        }
        // finally do the normalizing routine to complete the average pool
        // last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        let normalizer = atten_mask.sum_axis(Axis(1));
        let res = last_hidden_state.sum_axis(Axis(1));
        let mut result: Vec<Vec<f32>> = Vec::new();
        for idx_r in 0..res.shape()[0] {
            let embedds = res
                .slice(s![idx_r..idx_r + 1, ..])
                .iter()
                .map(|&v| v / normalizer[[idx_r]] as f32)
                .collect::<Vec<f32>>();
            result.push(embedds);
        }
        Ok(result)
    }
    pub fn encode(&self, text: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut context_encodings: Vec<tokenizers::Encoding> = Vec::new();
        let mut max_len: usize = 0;
        for i in 0..text.len() {
            let context_encoding;
            match self.tokenizer.encode(&*text[i], true) {
                Ok(res) => context_encoding = res,
                Err(err) => {
                    bail!("unable to tokenize the input: {}", err)
                }
            }
            max_len = max(max_len, context_encoding.get_attention_mask().len());
            max_len = max(max_len, context_encoding.get_ids().len());
            max_len = max(max_len, context_encoding.get_type_ids().len());
            context_encodings.push(context_encoding)
        }
        let mut ids_arr = Array::<i64, _>::zeros((context_encodings.len(), max_len));
        let mut type_ids_arr = Array::<i64, _>::zeros((context_encodings.len(), max_len));
        let mut atten_mask_arr = Array::<i64, _>::zeros((context_encodings.len(), max_len));

        for idx in 0..context_encodings.len() {
            let type_ids: Vec<i64> = context_encodings[idx]
                .get_type_ids()
                .iter()
                .map(|&p| p as i64)
                .collect::<Vec<i64>>();
            let ids: Vec<i64> = context_encodings[idx]
                .get_ids()
                .iter()
                .map(|&p| p as i64)
                .collect::<Vec<i64>>();
            let atten_mask: Vec<i64> = context_encodings[idx]
                .get_attention_mask()
                .iter()
                .map(|&p| p as i64)
                .collect::<Vec<i64>>();

            ids_arr
                .slice_mut(s![idx..idx + 1, 0..ids.len()])
                .assign(&ArrayView::from(&ids));
            type_ids_arr
                .slice_mut(s![idx..idx + 1, 0..type_ids.len()])
                .assign(&ArrayView::from(&type_ids));
            atten_mask_arr
                .slice_mut(s![idx..idx + 1, 0..atten_mask.len()])
                .assign(&ArrayView::from(&atten_mask));
        }
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self.session.run([
            InputTensor::from_array(ids_arr.into_dyn()),
            InputTensor::from_array(atten_mask_arr.clone().into_dyn()),
            InputTensor::from_array(type_ids_arr.into_dyn()),
        ])?;

        let last_hidden: OrtOwnedTensor<f32, _> = outputs[0].try_extract().unwrap();
        let last_hidden = last_hidden.view().to_owned();
        self.average_pool(last_hidden, atten_mask_arr.clone())
    }

    pub fn new(params: &EmbedderParams) -> anyhow::Result<Self> {
        let model_details;
        match E5_MODELS.get(params.model_name) {
            Some(d) => model_details = d,
            None => {
                bail!("E5_MODEL: {} was not found", params.model_name)
            }
        }
        if !params.model_path.exists() {
            fs::create_dir_all(params.model_path)?;
        }
        let model_file_path = PathBuf::from(params.model_path).join(model_details.filename);
        if !model_file_path.exists() {
            info!(
                model = params.model_name,
                "model_file_path: {:?} does not exist - initiating download", model_file_path
            );
            download_model_sync(
                params.model_name,
                &format!("{}/e5/{}", S3_BUCKET_URI, model_details.filename),
                true,
                &model_file_path,
                model_details.hash_val,
            )?;
        }
        let session = SessionBuilder::new(&params.ort_environment)?
            .with_inter_threads(params.num_threads)?
            .with_parallel_execution(params.parallel_execution)?
            .with_model_from_file(model_file_path)?;
        let tokenizer: Tokenizer;
        match Tokenizer::from_pretrained(model_details.tokenizer_name, None) {
            Ok(tk) => tokenizer = tk,
            Err(err) => {
                bail!("unable to create a tokenizer: {}", err);
            }
        }
        Ok(E5Embedder {
            session: session,
            tokenizer: tokenizer,
        })
    }
}
