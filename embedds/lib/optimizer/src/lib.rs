use itertools::iproduct;
use itertools::Itertools;
use ndarray::{s, Array, ArrayView};
use std::collections::{HashMap, HashSet};
use tch::Tensor;
use tracing::info;

extern crate serde;
extern crate serde_json;

use serde::Deserialize;

pub struct Optimizer {}

#[derive(Clone, Debug)]
#[allow(unused)]
pub struct AccuracyEstimate {
    accuracy: f32,
    standard_error: f32,
}
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct OptimizerResult {
    train_acc_no_matrix: AccuracyEstimate,
    test_acc_no_matrix: AccuracyEstimate,
    best_w_matrix: AccuracyEstimate,
    best_learned_matrix: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct EmbeddingPair {
    eid_1: String,
    eid_1_embedds: Vec<f32>,
    eid_2: String,
    eid_2_embedds: Vec<f32>,
    label: u8,
}

#[derive(Clone, Hash, Eq, PartialEq, Debug)]
struct Pair {
    eid_1: String,
    eid_2: String,
}

#[derive(Debug, Deserialize)]
#[allow(unused)]
pub struct EmbeddingRecord {
    #[serde(default)]
    sentence: String,
    #[serde(default)]
    embeddings: Vec<f32>,
}

pub fn from<'a, T>(t: &'a Tensor) -> T
where
    <T as TryFrom<&'a tch::Tensor>>::Error: std::fmt::Debug,
    T: TryFrom<&'a Tensor>,
{
    T::try_from(t).unwrap()
}
pub fn vec_f32_from(t: &Tensor) -> Vec<f32> {
    from::<Vec<f32>>(&t.reshape(&[-1]))
}

impl Optimizer {
    pub fn new() -> Self {
        return Optimizer {};
    }

    fn get_tensors(&self, pairs: &[EmbeddingPair]) -> (Tensor, Tensor, Tensor) {
        let mut a = Array::<f32, _>::zeros((pairs.len(), pairs[0].eid_1_embedds.len()));
        let mut b = Array::<f32, _>::zeros((pairs.len(), pairs[0].eid_2_embedds.len()));
        let mut c = Array::<f32, _>::zeros((pairs.len(), 1));
        pairs.iter().enumerate().for_each(|(idx, s)| {
            a.slice_mut(s![idx..idx + 1, ..]).assign(
                &ArrayView::from_shape((1, s.eid_1_embedds.len()), &s.eid_1_embedds)
                    .expect("unable")
                    .to_owned(),
            );
            b.slice_mut(s![idx..idx + 1, ..]).assign(
                &ArrayView::from_shape((1, s.eid_2_embedds.len()), &s.eid_2_embedds)
                    .expect("unable")
                    .to_owned(),
            );
            if s.label == 1 {
                c[[idx, 0]] = 1.0
            } else {
                c[[idx, 0]] = -1.0
            }
        });
        let e1 = Tensor::try_from(a).unwrap();
        let e2 = Tensor::try_from(b).unwrap();
        let s = Tensor::try_from(c).unwrap();
        (e1, e2, s)
    }

    fn get_applied_similarities(&self, matrix: &Tensor, embeddings: &Tensor) -> Tensor {
        let embedding_pairs = embeddings.split(2048, 1);
        let e1 = &embedding_pairs[0];
        let e2 = &embedding_pairs[1];
        let modified_embedding_1 = e1.mm(&matrix);
        let modified_embedding_2 = e2.mm(&matrix);
        let similarity =
            tch::Tensor::cosine_similarity(&modified_embedding_1, &modified_embedding_2, 1, 1e-8);
        similarity
    }

    fn model(
        &self,
        dims: i64,
        dropout_fraction: f64,
        embeddings: &tch::Tensor,
        matrix: &tch::Tensor,
    ) -> Tensor {
        let embedding_pairs = embeddings.split(dims, 1);
        let e1 = embedding_pairs[0].dropout(dropout_fraction, true);
        let e2 = embedding_pairs[1].dropout(dropout_fraction, true);
        let modified_embedding_1 = e1.mm(&matrix);
        let modified_embedding_2 = e2.mm(&matrix);
        let similarity =
            tch::Tensor::cosine_similarity(&modified_embedding_1, &modified_embedding_2, 1, 1e-8);
        similarity
    }
    fn compute_raw_cosine_similarities(&self, dims: i64, embeddings: &tch::Tensor) -> Tensor {
        let embedding_pairs = embeddings.split(dims, 1);
        println!(
            "num_embedding_pairs: {:?} | embeddings_pairs: {:?}",
            embedding_pairs.len(),
            embedding_pairs[0].size()
        );
        let similarity =
            tch::Tensor::cosine_similarity(&embedding_pairs[0], &embedding_pairs[1], 1, 1e-8);
        similarity
    }

    fn mse_loss(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let num_elements: i64 = predictions.numel().try_into().unwrap();
        let diff = predictions.reshape(&[num_elements]) - targets.reshape(&[num_elements]);
        let res = (diff.square().sum(tch::kind::Kind::Float)) / (diff.numel() as f64);
        res
    }

    fn accuracy_and_stderr(
        &self,
        similarities: &Tensor,
        labeled_similarity: &Tensor,
    ) -> (f32, f32) {
        let slim_vec = vec_f32_from(similarities);
        let lab_vec = vec_f32_from(labeled_similarity);
        let mut max_accuracy: f32 = 0.0;
        for threshold_thousandths in -1000..1000 {
            let threshold: f32 = threshold_thousandths as f32 / 1000.0;
            let mut total = 1.0;
            let mut correct = 1.0;
            slim_vec.iter().zip(lab_vec.iter()).for_each(|(&cs, &ls)| {
                total += 1.0;
                let prediction = if cs > threshold { 1.0 } else { -1.0 };
                if prediction == ls {
                    correct += 1.0
                }
            });
            let accuracy = correct / total;
            if accuracy > max_accuracy {
                max_accuracy = accuracy;
            }
        }
        let a = max_accuracy;
        let n = slim_vec.len() as f32;
        let standard_error = (a * (1.0 - a) / n).powf(0.5); // standard error of binomial
        (a, standard_error * 1.96)
    }

    pub fn optimize_matrix(
        &self,
        batch_size: i64,
        learning_rate: f64,
        max_epochs: usize,
        dropout_fraction: f64,
        train_samples: &[EmbeddingPair],
        test_samples: &[EmbeddingPair],
    ) -> anyhow::Result<OptimizerResult> {
        // create tensors from the flat training and testing samples
        let (e1_train, e2_train, s_train) = self.get_tensors(train_samples);
        let (e1_test, e2_test, s_test) = self.get_tensors(test_samples);
        let dims: i64 = e1_train.size()[1].try_into()?;
        // we concat here since the tch::data::Iter2 only takes 2 tensors
        // as arguments - all operations after need to *split* along dim=1
        let embedds_train = tch::Tensor::concat(&[e1_train, e2_train], 1);
        let embedds_test = tch::Tensor::concat(&[e1_test, e2_test], 1);
        // create the random application matrix that we intend to use
        let mut matrix =
            tch::Tensor::rand(&[dims, dims], (tch::kind::Kind::Float, tch::Device::Cpu));
        matrix = matrix.set_requires_grad(true);
        // compute our base estimates of accuracy and the standard error
        let cs_similar_train = self.compute_raw_cosine_similarities(dims, &embedds_train);
        let (acc_train, se_train) = self.accuracy_and_stderr(&cs_similar_train, &s_train);
        info!("train - acc: {:?}, se: {:?}", acc_train, se_train);
        let cs_similar_test = self.compute_raw_cosine_similarities(dims.try_into()?, &embedds_test);
        let (acc_test, se_test) = self.accuracy_and_stderr(&cs_similar_test, &s_test);
        info!("test - acc: {:?}, se: {:?}", acc_test, se_test);

        // now run the computation for the specified epochs
        let mut best_acc_seen: f32 = f32::MIN;
        let mut best_stde_seen: f32 = f32::MIN;
        let mut best_learned_matrix: Vec<f32> = Vec::new();
        for epoch in 0..max_epochs {
            tch::data::Iter2::new(&embedds_train, &s_train, batch_size)
                .shuffle()
                .for_each(|(embedds, labels)| {
                    let predicted_similarity =
                        self.model(dims, dropout_fraction, &embedds, &matrix);
                    let loss = self.mse_loss(&predicted_similarity, &labels);
                    loss.backward();
                    tch::no_grad(|| {
                        matrix -= (matrix.grad()).multiply_scalar(learning_rate);
                        matrix.zero_grad();
                    });
                });
            // compute the test loss
            let test_predictions = self.model(dims, dropout_fraction, &embedds_test, &matrix);
            let test_loss = self.mse_loss(&test_predictions, &s_test);
            let predictions = self.get_applied_similarities(&matrix, &embedds_test);
            let (acc, se) = self.accuracy_and_stderr(&predictions, &s_test);
            info!(
                "epoch: {:?} acc: {:?} std: {:?} test_loss: {:?}",
                epoch, acc, se, test_loss
            );
            if acc > best_acc_seen {
                // update if we see a better accuracy returned
                info!(
                    "new acc best found at epoch={:?} -> prev: {:?} new: {:?}",
                    epoch, best_acc_seen, acc
                );
                best_acc_seen = acc;
                best_stde_seen = se;
                best_learned_matrix = vec_f32_from(&matrix);
            }
        }
        // return a summary of what happend
        Ok(OptimizerResult {
            train_acc_no_matrix: AccuracyEstimate {
                accuracy: acc_train,
                standard_error: se_train,
            },
            test_acc_no_matrix: AccuracyEstimate {
                accuracy: acc_test,
                standard_error: se_test,
            },
            best_w_matrix: AccuracyEstimate {
                accuracy: best_acc_seen,
                standard_error: best_stde_seen,
            },
            best_learned_matrix: best_learned_matrix,
        })
    }

    #[allow(unused)]
    fn generate_negatives(&self, pairs: &[EmbeddingPair]) -> Vec<EmbeddingPair> {
        // returns a list of negative_pairs by combining groups of positive pairs
        let mut embedds_by_eid: HashMap<String, Vec<f32>> = HashMap::new();
        let mut positives: Vec<EmbeddingPair> = Vec::new();
        let mut positive_pairs = HashSet::new();
        let mut batch_a: Vec<String> = Vec::new();
        let mut batch_b: Vec<String> = Vec::new();
        pairs.iter().for_each(|p| {
            if p.label == 1 {
                positives.push(p.clone());
                positive_pairs.insert(Pair {
                    eid_1: p.eid_1.clone(),
                    eid_2: p.eid_2.clone(),
                });
                batch_a.push(p.eid_1.clone());
                batch_b.push(p.eid_2.clone());
                embedds_by_eid.insert(p.eid_1.clone(), p.eid_1_embedds.clone());
                embedds_by_eid.insert(p.eid_2.clone(), p.eid_2_embedds.clone());
            }
        });
        let all_pairs: Vec<Pair> = iproduct!(batch_a.iter(), batch_b.iter())
            .map(|tuple| Pair {
                eid_1: tuple.0.to_string(),
                eid_2: tuple.1.to_string(),
            })
            .unique()
            .collect();
        let mut negative_pairs: Vec<EmbeddingPair> = Vec::new();
        all_pairs.iter().for_each(|p| {
            if !positive_pairs.contains(p) {
                negative_pairs.push(EmbeddingPair {
                    eid_1: p.eid_1.clone(),
                    eid_1_embedds: embedds_by_eid
                        .get(&p.eid_1)
                        .expect("unable to get the eids_1_embedds")
                        .to_vec(),
                    eid_2: p.eid_2.clone(),
                    eid_2_embedds: embedds_by_eid
                        .get(&p.eid_2)
                        .expect("unable to get the eids_1_embedds")
                        .to_vec(),
                    label: 0,
                });
            }
        });
        negative_pairs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use rand::prelude::SliceRandom;
    // use rand::{seq::IteratorRandom, thread_rng};
    use rand_split::train_test_split;
    use std::fs::File;
    use std::io::{prelude::*, BufReader};
    use tracing::Level;

    fn read_snli_embeddings(
        path_to_snli: &str,
    ) -> anyhow::Result<HashMap<String, (EmbeddingRecord, i64)>> {
        let mut result = HashMap::new();
        let file = File::open(path_to_snli)?;
        let reader = BufReader::new(file);
        let mut num_lines_read = 0;
        let mut id_counter = 0;
        for line in reader.lines() {
            if num_lines_read == 0 {
                num_lines_read += 1;
                continue;
            }
            let l = line?;
            let r: Result<EmbeddingRecord, _> = serde_json::from_str(&l);
            match r {
                Ok(res) => {
                    result.insert(res.sentence.clone(), (res, id_counter));
                    id_counter += 1;
                }
                Err(err) => return Err(err.into()),
            }
        }
        Ok(result)
    }
    fn prepare_optimizing_lib(
        embedds_by_sentence: HashMap<String, (EmbeddingRecord, i64)>,
        path_to_snli_labels: &str,
    ) -> anyhow::Result<Vec<EmbeddingPair>> {
        let file = File::open(path_to_snli_labels)?;
        let reader = BufReader::new(file);
        let mut num_lines_read = 0;
        // let num_pairs_to_consider = 1000;
        let mut num_positive = 1000;
        let mut num_negaitve = 1000;

        let mut embeding_pairs: Vec<EmbeddingPair> =
            Vec::with_capacity(num_positive + num_negaitve);
        for line in reader.lines() {
            if num_positive == 0 && num_negaitve == 0 {
                break;
            }
            let l = line?;
            let parts: Vec<&str> = l.split("\t").collect();
            if num_lines_read != 0 {
                let label = parts[0];
                let sentence_1 = parts[5];
                let sentence_2 = parts[6];
                if label != "entailment" && label != "contradiction" {
                    continue;
                }
                let item_id_1 = embedds_by_sentence.get(sentence_1);
                let item_id_2 = embedds_by_sentence.get(sentence_2);
                if item_id_1.is_none() || item_id_2.is_none() {
                    continue;
                }
                let item_1_details = item_id_1
                    .ok_or_else(|| anyhow::anyhow!("unwrap of option unexpectedly failed"))?;
                let item_2_details = item_id_2
                    .ok_or_else(|| anyhow::anyhow!("unwrap of option unexpectedly failed"))?;

                let item_label = if label == "entailment" { 1 } else { 0 };
                if item_label == 1 {
                    if num_positive == 0 {
                        continue;
                    }
                    num_positive -= 1
                } else {
                    if num_negaitve == 0 {
                        continue;
                    }
                    num_negaitve -= 1
                }
                embeding_pairs.push(EmbeddingPair {
                    eid_1: item_1_details.1.to_string(),
                    eid_1_embedds: item_1_details.0.embeddings.clone(),
                    eid_2: item_2_details.1.to_string(),
                    eid_2_embedds: item_2_details.0.embeddings.clone(),
                    label: if label == "entailment" { 1 } else { 0 },
                });
            }

            num_lines_read += 1;
        }
        Ok(embeding_pairs)
    }

    #[test]
    fn test_optimizer() {
        // instantiate the subscribers so we can see what
        // is going on!
        let subscriber = tracing_subscriber::fmt()
            .with_max_level(Level::INFO)
            .finish();
        match tracing::subscriber::set_global_default(subscriber) {
            Ok(()) => {}
            Err(err) => {
                panic!("unable to create the tracing subscriber: {}", err)
            }
        }
        // generate the embeddings that we care about from the file
        // this follows the openai example to make things comparable
        let path_to_snli_embedds = "../../.cache/snli_embedding.csv";
        let path_to_snli_labels = "../.data/snli_1.0/snli_1.0_train.txt";
        // get a mapping of sentence -> (embedding, id)
        let embedds_by_sentence = read_snli_embeddings(path_to_snli_embedds)
            .expect("unable to build the sentence -> (embedding, id) map");
        // build a mapping of the pairs that we care about
        let mut embedding_pairs = prepare_optimizing_lib(embedds_by_sentence, path_to_snli_labels)
            .expect("unable to generate the embedding_pairs");

        // now split things out into segments
        let mut train_samples: Vec<EmbeddingPair> = Vec::new();
        let mut test_samples: Vec<EmbeddingPair> = Vec::new();
        train_test_split(&mut embedding_pairs, 0.5, 0.5)
            .iter()
            .enumerate()
            .for_each(|(idx, s)| {
                if idx == 0 {
                    train_samples.extend_from_slice(s);
                } else {
                    test_samples.extend_from_slice(s);
                }
            });

        let optimizer = Optimizer::new();
        // embedding_pairs.iter().enumerate().for_each(|(idx, p)| {
        //     if idx < 900 {
        //         train_samples.push(p.clone())
        //     } else {
        //         test_samples.push(p.clone())
        //     }
        // });
        // run the training!
        let max_epochs = 30;
        let dropout_fraction: f64 = 0.2;
        let pairs: Vec<(i64, f64)> = vec![(10, 10.0), (100, 100.0)];
        pairs.iter().for_each(|p| {
            let batch_size: i64 = p.0;
            let learning_rate: f64 = p.1;
            optimizer
                .optimize_matrix(
                    batch_size,
                    learning_rate,
                    max_epochs,
                    dropout_fraction,
                    &train_samples,
                    &test_samples,
                )
                .expect("unable to run the optimization process");
        });
    }
}
