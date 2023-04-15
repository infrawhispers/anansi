use std::fs;
use std::path::Path;
use std::path::PathBuf;

use std::sync::Arc;

use anyhow::bail;
use ndarray::{Array, ArrayView};
use ort::{
    environment::Environment, tensor::DynOrtTensor, tensor::FromArray, tensor::InputTensor,
    tensor::OrtOwnedTensor, ExecutionProvider, GraphOptimizationLevel, Session, SessionBuilder,
};
use phf::phf_map;
use tokio::runtime::Runtime;

use tokenizers::tokenizer::Tokenizer;

use crate::embedder;
use crate::embedder::Embedder;
use crate::utils::download_model_sync;

pub struct CLIPEmbedder {
    session_visual: Session,
    session_textual: Session,
    tokenizer: Tokenizer,
}

#[derive(Clone, Copy)]
struct CLIPModel {
    pub textual: &'static str,
    pub textual_hash: &'static str,
    pub visual: &'static str,
    pub visual_hash: &'static str,
}
static S3_BUCKET_V2: &'static str = "https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/";
static CLIP_MODELS: phf::Map<&'static str, CLIPModel> = phf_map! {
    "RN50_OPENAI" => CLIPModel{
        textual: "RN50/textual.onnx",
        textual_hash:"722418bfe47a1f5c79d1f44884bb3103",
        visual: "RN50/visual.onnx",
        visual_hash: "5761475db01c3abb68a5a805662dcd10",
    },
    "RN50_YFCC15M" => CLIPModel{
        textual: "RN50-yfcc15m/textual.onnx",
        textual_hash: "4ff2ea7228b9d2337b5440d1955c2108",
        visual: "RN50-yfcc15m/visual.onnx",
        visual_hash: "87daa9b4a67449b5390a9a73b8c15772"
    },
    "RN50_CC12M" => CLIPModel{
        textual: "RN50-cc12m/textual.onnx",
        textual_hash: "78fa0ae0ea47aca4b8864f709c48dcec",
        visual: "RN50-cc12m/visual.onnx",
        visual_hash: "0e04bf92f3c181deea2944e322ebee77",
    },
    "CLIP_VIT_L_14_336_OPENAI" => CLIPModel {
        textual: "ViT-L-14@336px/textual.onnx",
        textual_hash: "78fab479f136403eed0db46f3e9e7ed2",
        visual: "ViT-L-14@336px/visual.onnx",
        visual_hash: "f3b1f5d55ca08d43d749e11f7e4ba27e",
    },
};

impl Embedder for CLIPEmbedder {
    fn new(params: &embedder::EmbedderParams) -> anyhow::Result<Self> {
        CLIPEmbedder::new(params)
    }
    fn encode(&self, req: &embedder::EmebeddingRequest) -> anyhow::Result<Vec<Vec<f32>>> {
        let req_clip: &embedder::CLIPParams = match req {
            embedder::EmebeddingRequest::CLIPRequest { params } => &params,
            _ => {
                unreachable!("incorrect params passed for construction")
            }
        };
        match req_clip {
            embedder::CLIPParams::Text { vals } => {
                let mut res = Vec::with_capacity(vals.len());
                for i in 0..vals.len() {
                    let intermed = self.encode(&vals[i])?;
                    res.push(intermed)
                }
                return Ok(res);
            }
            embedder::CLIPParams::Uri { vals } => {
                bail!("this is not implemented as yet...")
            }
            embedder::CLIPParams::UriBytes { vals } => {
                bail!("this is not implemented as yet...")
            }
        }
    }
}
impl CLIPEmbedder {
    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        // we need to handle image resizing + handling!
        let text_encoding;
        match self.tokenizer.encode(text, true) {
            Ok(res) => text_encoding = res,
            Err(err) => {
                bail!("unable to tokenize the input: {}", err)
            }
        }
        // TODO(infrawhispers) context_length is expected to be 77 by the ONNX models served by JinaAI here:
        // https://github.com/jina-ai/clip-as-service/blob/1888ef65f20a94b38f318696e663d447c7cb1dc6/server/clip_server/model/tokenization.py
        // if we end up allowing for the chinese + multilanguage models, this code will need to change.
        let context_length: usize = 77;
        let mut i_tvals: Vec<i32> = text_encoding
            .get_ids()
            .iter()
            .cloned()
            .map(|x| x as i32)
            .collect();
        let i_tvals_diff = context_length - i_tvals.len();
        if i_tvals_diff > 0 {
            i_tvals.extend(std::iter::repeat(0).take(i_tvals_diff));
        }
        let a: Array<i32, _> = ArrayView::from_shape((1, i_tvals.len()), &i_tvals)?.to_owned();

        let mut i_atten_mask: Vec<i32> = text_encoding
            .get_attention_mask()
            .iter()
            .cloned()
            .map(|x| x as i32)
            .collect();
        let i_atten_mask_diff = context_length - i_atten_mask.len();
        if i_atten_mask_diff > 0 {
            i_atten_mask.extend(std::iter::repeat(0).take(i_atten_mask_diff));
        }

        let b: Array<i32, _> =
            ArrayView::from_shape((1, i_atten_mask.len()), &i_atten_mask)?.to_owned();
        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> =
            self.session_textual.run([
                InputTensor::from_array(a.into_dyn()),
                InputTensor::from_array(b.into_dyn()),
            ])?;

        let embedding: OrtOwnedTensor<f32, _> = outputs[outputs.len() - 1].try_extract().unwrap();
        let embedding = embedding.view();
        let result: Vec<f32> = embedding.to_owned().iter().cloned().collect::<Vec<_>>();
        Ok(result)
    }
    pub fn new(params: &embedder::EmbedderParams) -> anyhow::Result<Self> {
        let model_details: CLIPModel;
        match CLIP_MODELS.get(params.model_name) {
            Some(m) => model_details = *m,
            None => {
                bail!(
                    "CLIP model: {} was not found; below is a list of all available models...",
                    params.model_name
                );
            }
        };
        if !params.model_path.exists() {
            fs::create_dir_all(params.model_path)?;
        }
        let text_model_fp = PathBuf::from(params.model_path).join("textual.onnx");
        if !text_model_fp.exists() {
            download_model_sync(
                &format!("{}.{}", params.model_name, "textual.onnx"),
                &format!("{}{}", S3_BUCKET_V2, model_details.textual),
                true,
                &text_model_fp,
                model_details.textual_hash,
            )?;
        }
        let visual_model_fp = PathBuf::from(params.model_path).join("visual.onnx");
        if !visual_model_fp.exists() {
            download_model_sync(
                &format!("{}.{}", params.model_name, "visual.onnx"),
                &format!("{}{}", S3_BUCKET_V2, model_details.visual),
                true,
                &visual_model_fp,
                model_details.visual_hash,
            )?;
        }
        let session_textual = SessionBuilder::new(&params.ort_environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_inter_threads(params.num_threads)?
            .with_parallel_execution(true)?
            .with_model_from_file(text_model_fp)?;
        let session_visual = SessionBuilder::new(&params.ort_environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_inter_threads(params.num_threads)?
            .with_parallel_execution(true)?
            .with_model_from_file(visual_model_fp)?;
        let tokenizer: Tokenizer;
        match Tokenizer::from_pretrained("openai/clip-vit-base-patch16", None) {
            Ok(tk) => tokenizer = tk,
            Err(err) => {
                bail!("unable to create a tokenizer: {}", err);
            }
        }
        Ok(CLIPEmbedder {
            session_textual: session_textual,
            session_visual: session_visual,
            tokenizer: tokenizer,
        })
    }
}
