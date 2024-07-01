use anyhow::Result;
use arrow::{
    array::{Float32Array, RecordBatch},
    datatypes::{DataType, Field, Fields, Schema},
};
use arrow_udf::function;
use async_stream::stream;
use async_trait::async_trait;
use datafusion::{common::tree_node::TreeNodeIterator, prelude::*};
use futures_util::Stream;
use futures_util::StreamExt;
use std::{pin::Pin, sync::Arc};

#[function("mul(float32, float32) -> float32", output = "eval_mul_f32")]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}

pub trait Apply {
    fn apply<F, T>(&self, f: F) -> T
    where
        F: Fn(&RecordBatch) -> T;
}

impl Apply for RecordBatch {
    fn apply<F, T>(&self, f: F) -> T
    where
        F: Fn(&RecordBatch) -> T,
    {
        f(self)
    }
}

pub trait ApplyVec {
    fn apply<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&RecordBatch) -> T;
}
impl ApplyVec for Vec<RecordBatch> {
    fn apply<F, T>(&self, f: F) -> Vec<T>
    where
        F: Fn(&RecordBatch) -> T,
    {
        self.iter().map(|rb| f(rb)).collect::<Vec<T>>()
    }
}

#[async_trait]
pub trait ApplyDf {
    async fn apply<F, T>(self, f: F) -> Pin<Box<dyn Stream<Item = T> + Send>>
    where
        F: Fn(&RecordBatch) -> T + Send + Sync + 'static,
        T: Send + 'static;
}

#[async_trait]
impl ApplyDf for DataFrame {
    async fn apply<F, T>(self, f: F) -> Pin<Box<dyn Stream<Item = T> + Send>>
    where
        F: Fn(&RecordBatch) -> T + Send + Sync + 'static,
        T: Send + 'static,
    {
        let s = stream! {
            let mut stream = self.execute_stream().await.unwrap();

            while let Some(batch) = stream.next().await {
                let result = f(&batch.unwrap());
                yield result;
            }
        };

        Box::pin(s)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let schema = Arc::new(Schema::new(Fields::from(vec![
        Field::new("a", DataType::Float32, true),
        Field::new("b", DataType::Float32, true),
    ])));

    let a = Arc::new(Float32Array::from_iter_values(vec![1.0, 2.0, 3.0]));
    let b = Arc::new(Float32Array::from_iter_values(vec![10.0, 10.0, 10.0]));

    let rb = RecordBatch::try_new(schema.clone(), vec![a, b]).unwrap();

    println!("Testing apply/lambda with RecordBatch");
    let res = rb.apply(eval_mul_f32).unwrap();

    println!("a, b = {:?}", rb);
    println!("a * b = {:?}", res);

    let sc = SessionContext::new();
    sc.register_batch("my_table", rb)?;

    let mut res = sc
        .sql("select a,b from my_table")
        .await
        .unwrap()
        .apply(eval_mul_f32)
        .await;

    println!("Testing apply/lambda with dataframe");

    while let Some(z) = res.next().await {
        println!("{:?}", z.unwrap());
    }

    let coll = sc
        .sql("select a,b from my_table")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let coll = coll.apply(eval_mul_f32);

    Ok(())
}
