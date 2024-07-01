use anyhow::Result;
use arrow::{
    array::{Float32Array, RecordBatch},
    datatypes::{DataType, Field, Fields, Schema},
};
use arrow_udf::function;
use async_stream::stream;
use async_trait::async_trait;
use datafusion::prelude::*;
use futures_util::Stream;
use futures_util::StreamExt;
use std::{pin::Pin, sync::Arc};

#[function("gcd(float32, float32) -> float32", output = "eval_mul_f32")]
fn mul(a: f32, b: f32) -> f32 {
    a * b
}

#[async_trait]
pub trait Apply {
    async fn apply<F, T>(&self, f: F) -> T
    where
        F: Fn(&RecordBatch) -> T + Send;
}

#[async_trait]
impl Apply for RecordBatch {
    async fn apply<F, T>(&self, f: F) -> T
    where
        F: Fn(&RecordBatch) -> T + Send,
    {
        f(self)
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
    let res = rb.apply(eval_mul_f32).await.unwrap();

    println!("a, b = {:?}", rb);
    println!("a * b = {:?}", res);

    let sc = SessionContext::new();
    sc.register_batch("my_table", rb)?;

    let df = sc.sql("select a,b from my_table").await.unwrap();

    println!("Testing apply/lambda with dataframe");
    let mut res = df.apply(eval_mul_f32).await;

    while let Some(z) = res.next().await {
        println!("{:?}", z.unwrap());
    }

    Ok(())
}
