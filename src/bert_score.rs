use std::any::Any;
use std::ops::Deref;
use std::sync::Arc;

use numpy::IntoPyArray;
use numpy::ndarray::ArrayView2;
use once_cell::sync::Lazy;
use pyo3::prelude::*;
use tch::{CModule, Device, IValue, no_grad_guard, NoGradGuard, Tensor};

pub static MODEL: Lazy<Arc<DebertaModel>> =
    Lazy::new(|| Arc::new(DebertaModel::from_pt("../deberta_trace.pt")));

pub struct DebertaModel {
    model: CModule,
    device: Device,
    guard: NoGradGuard,
}

fn cos_sim_per_batch(pred: &Tensor, gold: &Tensor) -> Tensor {
    // pred and gold are shape [batch_size, num_tokens, hidden_dim], and we want
    // to take the dot products of each token from pred and gold at each token idx

    // the typical .dot() operator only works on 1D tensors, but this can be done by
    // performing the operations of a dot product on the last dimension by multiplying
    // pred and gold element-wise, then summing that vector
    let dotted = (pred * gold).sum_dim_intlist(&[-1][..], false, tch::Kind::Float);
    // dotted is now shape [batch_size, num_tokens] because we've collapsed the last dimension
    // with floats due to the dot products

    // since cos_sim is the dot product divided by the L2 norms of the two vectors, we take
    // the 2-norm (p = 2) along the last dimension for these two vectors
    let pred_norm = pred.norm_scalaropt_dim(2, [-1], false);
    let gold_norm = gold.norm_scalaropt_dim(2, [-1], false);
    // these norms are also shape [batch_size, num_tokens] because we've once again collapsed
    // the last dimension in to norms, vec -> norm

    // this result is shape [batch_size, num_tokens] where each element is the cos_sim per token_idx
    // where token_idx is in range 0..num_tokens. It's a vector of per-token cos sim vectors.
    // Also, to avoid NaNs, clamp the denominator to some small number
    let cos_sim_per_token_idx = dotted / (pred_norm * gold_norm).clamp_min(1e-9);

    // Finally, get the mean of those cos sims per token to get a cos sim per sequence, so
    // a flat vector of avg cos sim between tokens of each token idx per batcb
    cos_sim_per_token_idx.mean_dim(-1, false, tch::Kind::Float)
}

impl DebertaModel {
    pub fn from_pt(path: &str) -> Self {
        Self {
            model: CModule::load(path).expect("Failed to load model"),
            device: Device::cuda_if_available(),
            guard: no_grad_guard(),
        }
    }
    pub fn get_last_hidden_state(
        &self,
        input_ids: ArrayView2<'_, i64>,
        attention_mask: ArrayView2<'_, i64>,
    ) -> Tensor {
        let (batch_size, seq_len) = input_ids.dim();

        let input_ids_tensor = Tensor::f_from_slice(input_ids.as_slice().unwrap())
            .unwrap()
            .reshape(&[batch_size as i64, seq_len as i64])
            .to_kind(tch::Kind::Int64)
            .to_device(self.device);

        let attention_mask_tensor = Tensor::f_from_slice(attention_mask.as_slice().unwrap())
            .unwrap()
            .reshape(&[batch_size as i64, seq_len as i64])
            .to_kind(tch::Kind::Int64)
            .to_device(self.device);

        let output = self
            .model
            .forward_is(&vec![
                IValue::from(input_ids_tensor),
                IValue::from(attention_mask_tensor),
            ])
            .expect("Error encountered during forward pass for model");

        if let IValue::Tensor(last_hidden) = output {
            last_hidden.copy()
        } else {
            panic!("Expected Tensor in position 0");
        }
    }

    pub async fn bert_score(
        &self,
        pred_input_ids: ArrayView2<'_, i64>,
        pred_attention_mask: ArrayView2<'_, i64>,
        gold_input_ids: ArrayView2<'_, i64>,
        gold_attention_mask: ArrayView2<'_, i64>,
    ) -> Vec<f64> {
        let pred_hidden_state: Tensor =
            self.get_last_hidden_state(pred_input_ids, pred_attention_mask);
        let gold_hidden_state: Tensor =
            self.get_last_hidden_state(gold_input_ids, gold_attention_mask);
        let cos_sim = cos_sim_per_batch(&pred_hidden_state, &gold_hidden_state);
        let result: Vec<f64> = cos_sim.try_into().unwrap();
        result
    }
}

#[cfg(test)]
mod tests {
    use numpy::PyArrayMethods;

    use super::*;

    #[test]
    fn test_batch_cos_sim_is_one_for_same_tensors() {
        let model = MODEL.clone();
        let pred_data = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ];

        let gold_data = [
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ];
        let pred = Tensor::f_from_slice(&pred_data.concat().concat())
            .unwrap()
            .reshape(&[3, 3, 3])
            .to_kind(tch::Kind::Float);
        let gold = Tensor::f_from_slice(&gold_data.concat().concat())
            .unwrap()
            .reshape(&[3, 3, 3])
            .to_kind(tch::Kind::Float);
        let cos_sim = cos_sim_per_batch(&pred, &gold);
        let res: Vec<f64> = cos_sim.try_into().unwrap();
        for cos_sim in res {
            assert!((1f64 - cos_sim).abs() < 0.0001)
        }
    }
}
#[test]
fn test_batch_cos_sim_is_zero() {
    let model = MODEL.clone();

    // Each cos sim between tokens will involve
    // a dot product of 1 * 0 + 1 * 1 + -1 * 1 = 0
    // as each vector of 3 numbers represents a token vector
    let pred_data = [
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    ];

    let gold_data = [
        [[0, 1, -1], [0, 1, -1], [0, 1, -1]],
        [[0, 1, -1], [0, 1, -1], [0, 1, -1]],
        [[0, 1, -1], [0, 1, -1], [0, 1, -1]],
    ];
    let pred = Tensor::f_from_slice(&pred_data.concat().concat())
        .unwrap()
        .reshape(&[3, 3, 3])
        .to_kind(tch::Kind::Float);
    let gold = Tensor::f_from_slice(&gold_data.concat().concat())
        .unwrap()
        .reshape(&[3, 3, 3])
        .to_kind(tch::Kind::Float);
    let cos_sim = cos_sim_per_batch(&pred, &gold);
    let res: Vec<f64> = cos_sim.try_into().unwrap();
    for cos_sim in res {
        assert!((0f64 - cos_sim).abs() < 0.0001)
    }
}
