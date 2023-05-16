use phf::phf_map;

#[derive(Clone, Copy)]
pub(crate) struct CLIPModel {
    pub textual: &'static str,
    pub textual_hash: &'static str,
    pub visual: &'static str,
    pub visual_hash: &'static str,
    pub image_size: i32,
}

// MODEL_NAME corresponds 1:1 with the folders storing the different models
// hence all names should abide by the following:
// pulled from StackOverflow: https://stackoverflow.com/a/35352640
// Windows (FAT32, NTFS): Any Unicode except NUL, \, /, :, *, ?, ", <, >, |. Also, no space character at the start or end, and no period at the end.
// Mac(HFS, HFS+): Any valid Unicode except : or /
// Linux(ext[2-4]): Any byte except NUL or /

pub(crate) static CLIP_MODELS: phf::Map<&'static str, CLIPModel> = phf_map! {
    "RN50_OPENAI" => CLIPModel{
        textual: "RN50/textual.onnx",
        textual_hash:"722418bfe47a1f5c79d1f44884bb3103",
        visual: "RN50/visual.onnx",
        visual_hash: "5761475db01c3abb68a5a805662dcd10",
        image_size: 224,
    },
    "RN50_YFCC15M" => CLIPModel{
        textual: "RN50-yfcc15m/textual.onnx",
        textual_hash: "4ff2ea7228b9d2337b5440d1955c2108",
        visual: "RN50-yfcc15m/visual.onnx",
        visual_hash: "87daa9b4a67449b5390a9a73b8c15772",
        image_size: 224,
    },
    "RN50_CC12M" => CLIPModel{
        textual: "RN50-cc12m/textual.onnx",
        textual_hash: "78fa0ae0ea47aca4b8864f709c48dcec",
        visual: "RN50-cc12m/visual.onnx",
        visual_hash: "0e04bf92f3c181deea2944e322ebee77",
        image_size: 224,
    },
    "RN101_OPENAI" => CLIPModel {
        textual: "RN101/textual.onnx",
        textual_hash: "2d9efb7d184c0d68a369024cedfa97af",
        visual: "RN101/visual.onnx",
        visual_hash: "0297ebc773af312faab54f8b5a622d71",
        image_size: 224,
    },
    "RN101_YFCC15M" => CLIPModel {
        textual: "RN101-yfcc15m/textual.onnx",
        textual_hash: "7aa2a4e3d5b960998a397a6712389f08",
        visual: "RN101-yfcc15m/visual.onnx",
        visual_hash: "681a72dd91c9c79464947bf29b623cb4",
        image_size: 224,
    },
    "RN50X4_OPENAI" => CLIPModel {
        textual: "RN50x4/textual.onnx",
        textual_hash: "d9d63d3fe35fb14d4affaa2c4e284005",
        visual: "RN50x4/visual.onnx",
        visual_hash: "16afe1e35b85ad862e8bbdb12265c9cb",
        image_size: 288,
    },
    "RN50X16_OPENAI" => CLIPModel {
        textual: "RN50x16/textual.onnx",
        textual_hash: "1525785494ff5307cadc6bfa56db6274",
        visual: "RN50x16/visual.onnx",
        visual_hash: "2a293d9c3582f8abe29c9999e47d1091",
        image_size: 384,
    },
    "RN50X64_OPENAI" => CLIPModel {
        textual: "RN50x64/textual.onnx",
        textual_hash: "3ae8ade74578eb7a77506c11bfbfaf2c",
        visual: "RN50x64/visual.onnx",
        visual_hash: "1341f10b50b3aca6d2d5d13982cabcfc",
        image_size: 448,
    },
    "VIT_B_32_OPENAI" => CLIPModel {
        textual: "ViT-B-32/textual.onnx",
        textual_hash: "bd6d7871e8bb95f3cc83aff3398d7390",
        visual: "ViT-B-32/visual.onnx",
        visual_hash: "88c6f38e522269d6c04a85df18e6370c",
        image_size: 224,
    },
    "VIT_B_32_LAION2B_E16" => CLIPModel {
        textual: "ViT-B-32-laion2b_e16/textual.onnx",
        textual_hash: "aa6eac88fe77d21f337e806417957497",
        visual: "ViT-B-32-laion2b_e16/visual.onnx",
        visual_hash: "0cdc00a9dfad560153d40aced9df0c8f",
        image_size: 224,
    },
    "VIT_B_32_LAION400M_E31" => CLIPModel {
        textual: "ViT-B-32-laion400m_e31/textual.onnx",
        textual_hash: "832f417bf1b3f1ced8f9958eda71665c",
        visual: "ViT-B-32-laion400m_e31/visual.onnx",
        visual_hash: "62326b925ae342313d4cc99c2741b313",
        image_size: 224,
    },
    "VIT_B_32_LAION400m_E32" => CLIPModel {
        textual: "ViT-B-32-laion400m_e32/textual.onnx",
        textual_hash: "93284915937ba42a2b52ae8d3e5283a0",
        visual: "ViT-B-32-laion400m_e32/visual.onnx",
        visual_hash: "db220821a31fe9795fd8c2ba419078c5",
        image_size: 224,
    },
    "VIT_B_32_LAION2B_S34B_B79K" => CLIPModel {
        textual: "ViT-B-32-laion2b-s34b-b79k/textual.onnx",
        textual_hash: "84af5ae53da56464c76e67fe50fddbe9",
        visual: "ViT-B-32-laion2b-s34b-b79k/visual.onnx",
        visual_hash: "a2d4cbd1cf2632cd09ffce9b40bfd8bd",
        image_size: 224,
    },
    "VIT_B_16_OPENAI" => CLIPModel {
        textual: "ViT-B-16/textual.onnx",
        textual_hash: "6f0976629a446f95c0c8767658f12ebe",
        visual: "ViT-B-16/visual.onnx",
        visual_hash: "d5c03bfeef1abbd9bede54a8f6e1eaad",
        image_size: 224,
    },
    "VIT_B_16_LAION400M_E31" => CLIPModel {
        textual: "ViT-B-16-laion400m_e31/textual.onnx",
        textual_hash: "5db27763c06c06c727c90240264bf4f7",
        visual: "ViT-B-16-laion400m_e31/visual.onnx",
        visual_hash: "04a6a780d855a36eee03abca64cd5361",
        image_size: 224,
    },
    "VIT_B_16_LAION400M_E32" => CLIPModel {
        textual: "ViT-B-16-laion400m_e32/textual.onnx",
        textual_hash: "9abe000a51b6f1cbaac8fde601b16725",
        visual: "ViT-B-16-laion400m_e32/visual.onnx",
        visual_hash: "d38c144ac3ad7fbc1966f88ff8fa522f",
        image_size: 224,
    },
    "VIT_B_16_PLUS_240_LAION400M_E31" => CLIPModel {
        textual: "ViT-B-16-plus-240-laion400m_e31/textual.onnx",
        textual_hash: "2b524e7a530a98010cc7e57756937c5c",
        visual: "ViT-B-16-plus-240-laion400m_e31/visual.onnx",
        visual_hash: "a78989da3300fd0c398a9877dd26a9f1",
        image_size: 240,
    },
    "VIT_B_16_PLUS_240_LAION400M_E32" => CLIPModel {
        textual: "ViT-B-16-plus-240-laion400m_e32/textual.onnx",
        textual_hash: "53c8d26726b386ca0749207876482907",
        visual: "ViT-B-16-plus-240-laion400m_e32/visual.onnx",
        visual_hash: "7a32c4272c1ee46f734486570d81584b",
        image_size: 240,
    },
    "VIT_L_14_OPENAI" => CLIPModel {
        textual: "ViT-L-14/textual.onnx",
        textual_hash: "325380b31af4837c2e0d9aba2fad8e1b",
        visual: "ViT-L-14/visual.onnx",
        visual_hash: "53f5b319d3dc5d42572adea884e31056",
        image_size: 224,
    },
    "VIT_L_14_LAION400M_E31" => CLIPModel {
        textual: "ViT-L-14-laion400m_e31/textual.onnx",
        textual_hash: "36216b85e32668ea849730a54e1e09a4",
        visual: "ViT-L-14-laion400m_e31/visual.onnx",
        visual_hash: "15fa5a24916e2a58325c5cf70350c300",
        image_size: 224,
    },
    "VIT_L_14_LAION400M_E32"=> CLIPModel {
        textual: "ViT-L-14-laion400m_e32/textual.onnx",
        textual_hash: "8ba5b76ba71992923470c0261b10a67c",
        visual: "ViT-L-14-laion400m_e32/visual.onnx",
        visual_hash: "49db3ba92bd816001e932530ad92d76c",
        image_size: 224,
    },
    "VIT_L_14_LAION2B_S32B_B82K" => CLIPModel {
        textual: "ViT-L-14-laion2b-s32b-b82k/textual.onnx",
        textual_hash: "da36a6cbed4f56abf576fdea8b6fe2ee",
        visual: "ViT-L-14-laion2b-s32b-b82k/visual.onnx",
        visual_hash: "1e337a190abba6a8650237dfae4740b7",
        image_size: 224,
    },
    "VIT_L_14_336_OPENAI" => CLIPModel {
        textual: "ViT-L-14@336px/textual.onnx",
        textual_hash: "78fab479f136403eed0db46f3e9e7ed2",
        visual: "ViT-L-14@336px/visual.onnx",
        visual_hash: "f3b1f5d55ca08d43d749e11f7e4ba27e",
        image_size: 336
    },
    "VIT_H_14_LAION2B_S32B_B79K" => CLIPModel {
        textual: "ViT-H-14-laion2b-s32b-b79k/textual.onnx",
        textual_hash: "41e73c0c871d0e8e5d5e236f917f1ec3",
        visual: "ViT-H-14-laion2b-s32b-b79k/visual.zip",
        visual_hash: "38151ea5985d73de94520efef38db4e7",
        image_size: 224
    },
    "VIT_G_14_LAION2B_S12B_B42K" => CLIPModel {
        textual: "ViT-g-14-laion2b-s12b-b42k/textual.onnx",
        textual_hash: "e597b7ab4414ecd92f715d47e79a033f",
        visual: "ViT-g-14-laion2b-s12b-b42k/visual.zip",
        visual_hash: "6d0ac4329de9b02474f4752a5d16ba82",
        image_size: 224
    }
};
