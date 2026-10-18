# ModelServe — AI Inference and Monitoring Platform

A platform design for serving trained machine-learning models through FastAPI, managing model versions with MLflow, and recording prediction outcomes and operational metrics in PostgreSQL.

**Technology stack:** Python, FastAPI, PyTorch, Docker, AWS EC2, MLflow, PostgreSQL, GitHub Actions.

> This README describes the intended architecture. Adapt endpoints, commands, and folder names to your actual implementation. Performance results should be added only after measurement.

## Overview

Training a model does not automatically make it usable by another application. A deployed ML service also needs:

- A consistent input and output contract.
- Request validation.
- Reproducible preprocessing.
- Explicit model-version selection.
- Health checks.
- Prediction and failure logging.
- Repeatable deployment.
- Evaluation of predictive quality and serving performance.

ModelServe separates offline model development from online prediction serving.

MLflow records training experiments and model artifacts. FastAPI serves approved model versions. PostgreSQL stores operational records. Docker packages the service, and GitHub Actions coordinates testing and deployment to AWS EC2.

## Project Scope

The proposed initial implementation supports:

- One PyTorch classification model family.
- A fixed feature schema shared across supported versions.
- Single-record prediction requests.
- Default and explicit model-version selection.
- Preloaded approved model versions.
- PostgreSQL prediction and error logging.
- MLflow experiment tracking and model versioning.
- Docker deployment on an EC2 instance.
- Automated testing and deployment checks.

GPU scheduling, automatic retraining, autoscaling, and distributed model serving are future extensions.

## Architecture

### Online Prediction Path

```mermaid
flowchart TD
    Client["Client Application"] --> API["FastAPI Service"]
    API --> Validation["Request Validation"]
    Validation --> Catalog["Approved Model Catalog"]
    Catalog --> Preprocessing["Versioned Preprocessing"]
    Preprocessing --> Inference["PyTorch Inference"]
    Inference --> Response["Prediction Response"]
    Response --> Client
    API --> Logging["Outcome Logging"]
    Logging --> Database[("PostgreSQL")]
```

### Model Release and Deployment

```mermaid
flowchart TD
    Training["Train and Evaluate"] --> MLflow["MLflow Tracking"]
    MLflow --> Registry["Registered Model Artifacts"]
    Registry --> Approval["Compatibility and Approval Checks"]
    Approval --> Manifest["Pinned Release Manifest"]

    Commit["Application Commit"] --> CI["GitHub Actions Checks"]
    CI --> Image["Docker Image"]

    Image --> Deployment["EC2 Deployment"]
    Manifest --> Deployment
    Deployment --> Checks["Readiness and Smoke Tests"]
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| FastAPI | Validate requests, resolve versions, coordinate inference, and return responses |
| PyTorch | Run the loaded model in inference mode |
| Preprocessing bundle | Apply saved transformations and feature ordering |
| MLflow | Track experiments, metrics, artifacts, and registered versions |
| PostgreSQL | Store prediction outcomes, failures, and deployment metadata |
| Docker | Package application code and dependencies |
| AWS EC2 | Host the serving application |
| GitHub Actions | Run checks, build images, and coordinate deployment |

MLflow is used during model development and release preparation. Normal prediction requests should not download artifacts or query MLflow.

## End-to-End Prediction Flow

1. A client submits a JSON request.
2. FastAPI validates feature names, types, ranges, and missing values.
3. The service resolves the requested version or configured default.
4. Unknown or unapproved versions are rejected.
5. The service retrieves the already loaded model and preprocessing bundle.
6. Preprocessing transforms features in the training feature order.
7. The transformed values become a PyTorch tensor.
8. The model performs inference.
9. The service converts the output into the documented prediction format.
10. PostgreSQL records the model version, outcome, and timing metadata.
11. The API returns the prediction and request identifier.

## Example Use Case

A tabular churn classifier provides a concrete example.

This example illustrates the API contract; replace it with the actual model and features used in your implementation.

### Request

```json
{
  "model_version": "3",
  "features": {
    "tenure_months": 12,
    "monthly_spend": 799.0,
    "support_tickets_30d": 3,
    "usage_days_30d": 8,
    "payment_delays_90d": 1
  }
}
```

### Response

The following is an illustrative response, not a measured model result.

```json
{
  "request_id": "example-request-id",
  "model_name": "churn_classifier",
  "model_version": "3",
  "schema_version": "churn-v1",
  "predicted_label": 1,
  "positive_class_probability": 0.78
}
```

The probability represents the positive class according to the model's output contract. It is not a guarantee of correctness.

## API Endpoints

These are proposed endpoint conventions.

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Return a prediction |
| GET | `/models` | List approved model versions |
| GET | `/health/live` | Confirm that the application process is running |
| GET | `/health/ready` | Confirm that models and required dependencies are ready |

### Error Handling

| Condition | Suggested Status |
|---|---|
| Invalid or missing features | 422 |
| Unknown or unapproved model version | 404 |
| Approved model temporarily unavailable | 503 |
| Required database logging unavailable | 503 |
| Unexpected inference failure | 500 |

Error responses should include a request identifier and a clear error code without exposing stack traces, credentials, or internal paths.

## Input Validation

Validate inputs before preprocessing or inference.

Checks should include:

- Required feature names.
- Unexpected features.
- Numeric and categorical types.
- Missing-value policy.
- Allowed ranges.
- Maximum request size.
- Rejection of non-finite numeric values.
- Compatibility with the selected model's schema.

Validation prevents malformed requests from reaching the model. It does not establish whether valid-looking business data is accurate.

## Preprocessing

Training and inference must use the same transformations.

Each model version should include its saved preprocessing configuration, such as:

- Feature ordering.
- Scaling parameters.
- Category mappings.
- Missing-value handling.
- Expected tensor shape.
- Expected dtype.

Fit preprocessing only on training data, then save and reuse it.

Do not fit a scaler or encoder on incoming prediction requests.

## PyTorch Inference

The serving path should use both:

```python
model.eval()

with torch.inference_mode():
    output = model(input_tensor)
```

`model.eval()` changes the behaviour of layers such as dropout and batch normalization.

`torch.inference_mode()` disables gradient-related work during prediction.

The service must also apply the correct output interpretation:

- Sigmoid for a model trained to return binary logits.
- Softmax for a model trained to return multiclass logits.
- No repeated activation if the model already returns probabilities.

Document the class ordering and threshold.

## Model Versioning

Each approved model bundle should contain:

- Architecture reference.
- Trained weights.
- Saved preprocessing.
- Feature-schema identifier.
- Output class mapping.
- Decision threshold.
- MLflow run identifier.
- Dataset version or reference.
- Source commit.
- Dependency constraints.
- Artifact digest.
- Offline evaluation results.

### Release Manifest

Example:

```json
{
  "model_name": "churn_classifier",
  "default_version": "3",
  "approved_versions": ["2", "3"],
  "schema_version": "churn-v1",
  "application_commit": "replace-with-actual-commit",
  "image_digest": "replace-with-actual-image-digest",
  "model_artifact_digests": {
    "2": "replace-with-actual-digest",
    "3": "replace-with-actual-digest"
  }
}
```

Pin concrete versions and artifact digests for deployment.

Avoid resolving a moving label such as `latest` during every request.

## Model Loading

For the initial implementation:

1. Read the release manifest at startup.
2. Load only approved versions.
3. Verify artifact integrity.
4. Load preprocessing and model weights.
5. Run compatibility checks.
6. Perform a small smoke prediction.
7. Mark the service ready.

Keep the allowlist small enough to fit available memory.

Multiple application workers normally load separate model copies. Increasing worker count can increase memory consumption and CPU contention.

## MLflow Integration

Use MLflow to record:

- Training parameters.
- Dataset references.
- Model architecture configuration.
- Evaluation metrics.
- Training artifacts.
- Preprocessing artifacts.
- Registered model versions.

Example comparison:

| Run | Model Version | Dataset Reference | Metrics |
|---|---|---|---|
| Training run A | Version 2 | Recorded dataset version | Measured offline results |
| Training run B | Version 3 | Recorded dataset version | Measured offline results |

Model promotion should depend on documented acceptance criteria.

Passing application tests alone does not establish that a new model is better.

## PostgreSQL Logging

### Suggested Tables

| Table | Purpose |
|---|---|
| `model_releases` | Approved model metadata |
| `deployments` | Application and model deployment history |
| `prediction_logs` | Request outcomes and serving measurements |
| `ground_truth_labels` | Optional delayed labels for quality evaluation |

### Suggested Prediction Log Fields

```text
request_id
deployment_id
model_name
model_version
schema_version
status
predicted_label
positive_class_probability
inference_duration_ms
serving_duration_ms
error_code
created_at
```

Store only the information required for diagnosis and evaluation.

Raw feature payloads should not be logged by default.

### Timing Definitions

**Inference duration:** Model execution and output transformation.

**Serving duration:** Validation, version resolution, preprocessing, and inference, measured before recording the log.

**Client-observed latency:** Full round-trip time, including network and synchronous database logging.

Use a monotonic clock for durations and UTC timestamps for events.

### Logging Failure Policy

A simple initial design uses synchronous metadata logging.

If PostgreSQL is unavailable, return `503` rather than acknowledging a prediction that cannot be recorded.

This prioritizes traceability but makes database availability part of serving availability.

An asynchronous design is possible later, but it needs explicit buffering, durability, and backpressure behaviour.

## Monitoring

Stored records support queries and reports for:

- Request volume.
- Error rate.
- Latency percentiles.
- Prediction distributions.
- Version usage.
- Deployment comparisons.

| Signal | Meaning |
|---|---|
| Rising error rate | Possible serving or dependency problem |
| Rising latency | Possible load or resource contention |
| Changing label distribution | Output behaviour has changed |
| Changing probability distribution | Score behaviour has changed |
| Lower accuracy with ground truth | Predictive performance has degraded |

A change in prediction distribution does not prove accuracy degradation.

Online accuracy requires actual outcome labels.

PostgreSQL logging alone does not provide a live dashboard or alerting system.

## Docker Deployment

The serving image should include:

- Application code.
- Tested dependency versions.
- Runtime configuration conventions.
- Required model-loading code.
- A compatible approved artifact bundle or access to a verified local cache.

Persist PostgreSQL and MLflow data outside ephemeral container layers.

### Illustrative Commands

These commands require an implemented Dockerfile and application repository.

```bash
docker build -t modelserve:local .
```

```bash
docker run --rm \
  --env-file .env \
  -p 127.0.0.1:8000:8000 \
  modelserve:local
```

Database and MLflow addresses must be reachable from inside the container.

Inside a container, `localhost` refers to that container.

## AWS EC2 Deployment

A proposed release flow:

1. Build and test the Docker image.
2. Publish it to a configured container registry.
3. Record the immutable image digest.
4. Prepare a pinned model-release manifest.
5. Pull the image on EC2.
6. Start a candidate container.
7. Check readiness.
8. Run known-input smoke predictions.
9. Switch traffic after checks pass.
10. Retain the previous image and manifest for rollback.

Externally accessible deployments should use HTTPS and caller authentication.

PostgreSQL and MLflow should remain private.

A single EC2 instance remains a single point of failure.

## GitHub Actions CI/CD

### Pull-Request Checks

- Schema validation tests.
- Preprocessing tests.
- Inference tests with a small local fixture model.
- API tests.
- PostgreSQL integration tests.
- Docker build checks.
- Startup and readiness checks.

### Deployment Checks

- Pin application commit.
- Validate release manifest.
- Publish image by digest.
- Retrieve protected deployment credentials.
- Deploy candidate release.
- Run readiness and smoke tests.
- Restore the previous release if checks fail.

Application deployment and model promotion should remain explicit, separate decisions.

## Proposed Repository Structure

| Path | Responsibility |
|---|---|
| `app/main.py` | FastAPI application lifecycle |
| `app/api/predict.py` | Prediction endpoint |
| `app/api/health.py` | Health checks |
| `app/schemas/` | Request and response contracts |
| `app/services/model_catalog.py` | Approved version resolution |
| `app/services/inference.py` | Preprocessing and inference |
| `app/services/logging.py` | Outcome persistence |
| `app/db/` | Database models and sessions |
| `app/core/` | Configuration and error handling |
| `training/` | Training and MLflow tracking |
| `migrations/` | Database migrations |
| `tests/` | Automated tests |
| `benchmarks/` | Load tests and results |
| `deploy/` | Release manifests |
| `.github/workflows/` | CI/CD workflows |
| `Dockerfile` | Container definition |

This layout is a proposal; the listed files are not supplied by this README.

## Local Setup

The commands below become runnable after the corresponding application files exist.

### Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment Variables

```dotenv
DATABASE_URL=postgresql://modelserve:local_password@localhost:5432/modelserve
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_NAME=churn_classifier
MODEL_RELEASE_MANIFEST=./deploy/model-release.json
MODEL_CACHE_DIR=./model_cache
INFERENCE_DEVICE=cpu
LOG_LEVEL=INFO
```

Do not commit real credentials.

### Prepare Dependencies

1. Start PostgreSQL.
2. Apply the implemented migrations.
3. Configure MLflow tracking and artifact storage.
4. Train and evaluate the model.
5. Register an approved version.
6. Prepare the release manifest and model cache.

### Start the API

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Testing Strategy

### Model and Preprocessing Tests

- Verify feature ordering.
- Verify dtype and tensor dimensions.
- Compare saved preprocessing with training transformations.
- Check class ordering.
- Check finite outputs.
- Compare predictions with a reference batch.

### API Tests

- Valid default-version requests.
- Valid explicit-version requests.
- Missing and extra features.
- Invalid types.
- Out-of-range values.
- Unknown versions.
- Model-loading failures.
- Sanitized error responses.

### Database Tests

- Prediction logs contain the resolved version.
- Failure logs contain a useful error code.
- Request identifiers are consistently recorded.
- Database unavailability follows the documented policy.

### Deployment Tests

- Missing or corrupt artifacts fail readiness.
- Incompatible schemas fail startup checks.
- Candidate deployments pass known-input predictions.
- Rollback restores both application and model versions.

## Evaluation

### Offline Model Quality

Use a documented train, validation, and test split.

For classification, select appropriate metrics:

- Precision.
- Recall.
- F1.
- ROC-AUC.
- PR-AUC.

Report class balance, threshold, dataset size, and split strategy.

Fit preprocessing only on the training split.

### Serving Performance

Measure:

- Client-observed p50, p95, and p99 latency.
- Throughput.
- Error rate.
- CPU utilization.
- Peak memory.
- Startup time.

Record:

- EC2 instance type.
- CPU and RAM.
- Model size.
- Input dimensions.
- Worker count.
- PyTorch thread settings.
- Request concurrency.
- Database location.
- Logging policy.

Separate cold startup from warm prediction performance.

### Results Template

| Metric | Version A | Version B |
|---|---|---|
| Held-out F1 | Not measured | Not measured |
| Held-out PR-AUC | Not measured | Not measured |
| Warm p95 latency | Not measured | Not measured |
| Throughput | Not measured | Not measured |
| Error rate | Not measured | Not measured |
| Peak memory | Not measured | Not measured |
| Startup time | Not measured | Not measured |

Replace these entries only with measured results.

## Design Tradeoffs

| Decision | Benefit | Limitation |
|---|---|---|
| Preload approved models | Predictable warm inference | More startup time and memory |
| Fixed feature schema | Simpler compatibility | Schema changes need a new contract |
| Synchronous logging | Traceable acknowledged responses | Database failures affect availability |
| Single EC2 host | Simple initial deployment | Limited resilience |
| Versioned preprocessing | Consistent transformations | Requires compatibility checks |

## Future Improvements

- Authentication and rate limiting.
- Metrics dashboards and tested alerts.
- Managed database and shared artifact storage.
- Shadow or canary deployments.
- Delayed-label quality monitoring.
- Input drift checks.
- Request batching.
- GPU inference where justified.
- Autoscaling based on measured demand.

These are roadmap items, not completed functionality.

## Interview Explanation

ModelServe separates model development from online serving.

MLflow tracks experiments and registered artifacts. FastAPI loads approved models and their preprocessing bundles before accepting traffic. Each request is validated, transformed, and passed through PyTorch inference. PostgreSQL records the prediction outcome and resolved model version.

Docker and pinned release manifests make deployments reproducible. GitHub Actions runs checks and coordinates EC2 deployment.

Offline evaluation measures model quality, while request logs and load tests measure serving behaviour.
