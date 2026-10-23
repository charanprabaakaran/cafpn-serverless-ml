# CAFPN: Cache-Aware Serverless Framework for Real-Time ML in IoT Healthcare

A serverless machine learning inference system on AWS that reduces redundant model invocations using a DynamoDB-backed prediction cache with **range-based (bucketized) cache keys**. Built for heart disease prediction in IoT healthcare scenarios, where vitals change constantly and delays matter.

## Problem

Serverless platforms such as AWS Lambda suffer from cold starts and repeated model loading, which slow down predictions. In healthcare monitoring, inputs like heart rate or blood pressure change every second, so an exact-match cache would almost never hit. CAFPN (Cache-Aware Cloud Adaptive FaaS Processing Network) addresses this by grouping similar inputs into ranges before checking the cache.

## How It Works

1. Requests from the web frontend or IoT clients reach the backend through **Amazon API Gateway**, which exposes RESTful endpoints. A Flask web app receives the 13 clinical input fields.
2. **Bucketization:** continuous values are grouped into ranges (for example age in 5-year bands, resting blood pressure in 10 mmHg bands, cholesterol in 20 mg/dl bands, oldpeak rounded to the nearest 0.5).
3. A SHA-256 hash of the bucketed input becomes the cache key (`request_id`).
4. **Cache hit:** the stored prediction is read from **DynamoDB** and returned without calling Lambda.
5. **Cache miss:** the Flask app invokes the **AWS Lambda** inference function (ML model packaged as a container image in **Amazon ECR**), then stores the result in DynamoDB.

Two endpoints allow a direct comparison:

| Endpoint | Behaviour |
|---|---|
| `/predictsc` | With CAFPN caching (bucketized key, checks DynamoDB first) |
| `/normal` | Without caching (always invokes Lambda, still logs the result) |

## Tech Stack

| Area | Tools |
|---|---|
| Language | Python |
| Web app | Flask, HTML, CSS, JavaScript |
| AWS | Lambda, API Gateway, DynamoDB, ECR, CloudWatch, IAM (accessed with boto3) |
| Containerisation | Docker |
| ML | scikit-learn, XGBoost, CatBoost |
| Load testing | Locust |

## Machine Learning Models

Trained on the [Heart Disease UCI dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) as a binary classification task.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 0.90 | 0.90 | 0.90 | 0.90 |
| Gradient Boosting | 0.87 | 0.88 | 0.86 | 0.86 |
| CatBoost | 0.93 | 0.94 | 0.93 | 0.93 |
| Stacking | 0.93 | 0.94 | 0.93 | 0.93 |

The Stacking classifier uses CatBoost and Gradient Boosting as base models with XGBoost as the meta-classifier.

## Load Test Results

Locust, 10 concurrent users, 5 minutes. The cached test cycles through 15 fixed sample inputs, so these numbers represent a **repeated-input workload** where most requests become cache hits.

| Metric | Without caching | With CAFPN caching |
|---|---|---|
| Total requests | 1,840 | 5,403 |
| Requests per second | 6.13 | 18.01 |
| Failed requests | 0 | 0 |

Caching raised throughput roughly 3x for repeated inputs. Tail latency from cold starts and cache misses remained a challenge, so CAFPN is best viewed as a throughput and execution-efficiency optimisation, not a complete latency solution.

## Limitations and Future Work

- **Approximate matching:** bucketing means two slightly different inputs in the same range share one cached prediction. This is a trade-off between speed and precision that would need clinical validation for real use.
- Load test used a small set of repeated inputs, so real hit rates will be lower.
- Small public dataset and simple caching (no TTL or adaptive eviction yet).
- Tail-latency spikes under concurrent load.

**Next steps:** cache expiry (DynamoDB TTL), measuring real hit rates with varied inputs, Lambda provisioned concurrency, streaming IoT data, and measuring accuracy loss from bucketing.

## Project Structure

```
application.py        Flask app (cache logic, /predictsc and /normal endpoints)
database.py           Database helpers
pred_lambda/          Lambda inference function
templates/            HTML templates (form and result pages)
static/, assets/      Static files and images
locustfile.py         Load test script
requirements.txt      Python dependencies
stackingclassifier.sav  Trained stacking model
```

## Getting Started

```bash
git clone https://github.com/charanprabaakaran/cafpn-serverless-ml.git
cd cafpn-serverless-ml
pip install flask boto3 numpy scikit-learn joblib
python application.py
```

You need AWS credentials configured (for example with `aws configure` or an IAM role) and a deployed Lambda function and DynamoDB table. Update `LAMBDA_FUNCTION_NAME`, `DYNAMODB_TABLE`, and `REGION` in `application.py` to match your resources. The DynamoDB table uses `request_id` (string) as its partition key. Never commit AWS keys to the repository.

## Load Testing

```bash
locust -f locustfile.py --host http://localhost:5000
```

## Author

**Charan Prabaakaran**, MSc Cloud Computing
rvcharan131@gmail.com
