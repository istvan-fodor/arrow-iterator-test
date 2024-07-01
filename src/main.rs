use std::{
    ops::{Add, Rem},
    sync::Arc,
};

use arrow::{
    array::{Float32Array, Float32Builder, RecordBatch},
    datatypes::{DataType, Field, Fields, Schema},
};

use arrow_udf::function;

#[function("gcd(float32, float32) -> float32", output = "eval_mul_f32")]
fn mul(mut a: f32, mut b: f32) -> f32 {
    a * b
}

pub trait Apply {
    fn apply<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&RecordBatch) -> T;
}

// Implement the Apply trait for RecordBatch
impl Apply for RecordBatch {
    fn apply<F, T>(&self, f: F) -> T
    where
        F: FnOnce(&RecordBatch) -> T,
    {
        f(self)
    }
}

fn main() {
    let schema = Arc::new(Schema::new(Fields::from(vec![
        Field::new("a", DataType::Float32, true),
        Field::new("b", DataType::Float32, true),
    ])));

    let a = Arc::new(Float32Array::from_iter_values(vec![1.0, 2.0, 3.0]));
    let b = Arc::new(Float32Array::from_iter_values(vec![10.0, 10.0, 10.0]));

    let rb = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();

    let res = rb.apply(eval_mul_f32).unwrap();

    println!("a, b = {:?}", rb);
    println!("a * b = {:?}", res);
}
