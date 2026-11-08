const modelNames = {
  tfidf_ridge: "TF-IDF + Ridge",
  tfidf_sgd: "TF-IDF + SGD",
  tfidf_sgd_huber: "TF-IDF + Huber SGD",
  context_catboost: "Context CatBoost",
  context_catboost_shallow: "Context CatBoost shallow",
  text_context_ridge: "Text + context Ridge"
};

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Request failed (${response.status}): ${url}`);
  return response.json();
}

function renderTableRow(tableBody, values) {
  const row = document.createElement("tr");
  values.forEach((value, index) => {
    const cell = document.createElement(index === 0 ? "th" : "td");
    cell.textContent = value == null ? "n/a" : String(value);
    if (index === 0) cell.scope = "row";
    row.append(cell);
  });
  tableBody.append(row);
}

function renderHistogram(target, counts) {
  target.replaceChildren();
  const maximum = Math.max(...counts);
  counts.forEach((count, index) => {
    const bar = document.createElement("span");
    bar.style.height = `${Math.max(3, count / maximum * 100)}%`;
    bar.title = `${count.toLocaleString()} labels in bin ${index + 1}`;
    target.append(bar);
  });
}

function renderBars(target, values, labelKey, countKey) {
  target.replaceChildren();
  const maximum = Math.max(...values.map((item) => item[countKey]));
  values.forEach((item) => {
    const row = document.createElement("div");
    row.className = "category-row";
    const name = document.createElement("span");
    name.className = "category-name";
    name.textContent = item[labelKey];
    const track = document.createElement("div");
    track.className = "category-track";
    const fill = document.createElement("div");
    fill.className = "category-fill";
    fill.style.width = `${item[countKey] / maximum * 100}%`;
    track.append(fill);
    const count = document.createElement("b");
    count.className = "category-count";
    count.textContent = item[countKey].toLocaleString();
    row.append(name, track, count);
    target.append(row);
  });
}

function showError(error) {
  const node = document.querySelector("[data-error]");
  if (node) node.textContent = error.message;
}

async function loadDatasetPage() {
  const data = await getJson("/api/dataset");
  const full = data.full_train_eda;
  document.querySelector("#revision").textContent = data.revision;
  document.querySelector("#train-rows").textContent = data.split_rows.train.toLocaleString();
  document.querySelector("#test-rows").textContent = data.split_rows.test.toLocaleString();
  document.querySelector("#label-mean").textContent = data.target.mean.toFixed(3);
  document.querySelector("#label-sd").textContent = data.target.std_population.toFixed(3);
  document.querySelector("#label-median").textContent = data.target.quantiles["0.5"].toFixed(2);
  document.querySelector("#label-range").textContent = `${data.target.min.toFixed(2)} - ${data.target.max.toFixed(2)}`;
  document.querySelector("#title-missing").textContent = full.missing_or_blank.Title.toLocaleString();
  document.querySelector("#tag-missing").textContent = full.missing_or_blank.Alltags.toLocaleString();
  document.querySelector("#title-median").textContent = `${full.title_character_lengths.median} chars / ${full.title_word_counts.median} words`;
  document.querySelector("#tag-median").textContent = `${full.tag_token_counts.median} tokens`;
  document.querySelector("#date-range").textContent = `${full.postdate_range_utc.min.slice(0, 10)} to ${full.postdate_range_utc.max.slice(0, 10)}`;
  document.querySelector("#license-name").textContent = data.license?.name || "SMP Challenge Community License";
  if (data.sample_text_temporal_eda) {
    const sample = data.sample_text_temporal_eda;
    document.querySelector("#sample-eda-caption").textContent =
      `Seven-shard sample only: ${sample.rows.toLocaleString()} posts, ` +
      `median title ${sample.title_character_lengths.median} characters (${sample.title_word_counts.median} words), ` +
      `median ${sample.tag_token_counts.median} tag tokens, ${sample.missing_or_blank.Title.toLocaleString()} missing titles. Not full-train estimates.`;
  }

  renderHistogram(document.querySelector("#target-histogram"), data.target.histogram_counts);
  renderBars(
    document.querySelector("#category-bars"),
    full.category_frequencies,
    "category",
    "count"
  );

  const fields = document.querySelector("#field-schema");
  data.field_schema.forEach((field) => {
    const scope = field.name === "image" || field.name === "img_filepath"
      ? "Not read / not cached"
      : ["Uid", "Pid", "Longitude", "Latitude", "Geoaccuracy", "user_description", "location_description"].includes(field.name)
        ? "Audited or intentionally excluded"
        : "Available in release";
    renderTableRow(fields, [field.name, field.type, scope]);
  });

  const categoryTable = document.querySelector("#category-summary");
  full.category_frequencies.forEach((item) => {
    renderTableRow(categoryTable, [item.category, item.count.toLocaleString(), `${(100 * item.count / full.rows).toFixed(2)}%`]);
  });

  const inference = data.test_inference;
  document.querySelector("#test-inference-count").textContent = `${inference.prediction_rows.toLocaleString()} rows from ${inference.test_shards} shards`;
  document.querySelector("#test-inference-note").textContent = inference.test_metric_note;
  const testEda = inference.input_eda;
  document.querySelector("#test-input-rows").textContent = testEda.rows.toLocaleString();
  document.querySelector("#test-title-missing").textContent = testEda.missing_or_blank.Title.toLocaleString();
  document.querySelector("#test-tag-missing").textContent = testEda.missing_or_blank.Alltags.toLocaleString();
  document.querySelector("#test-date-range").textContent = `${testEda.postdate_range_utc.min.slice(0, 10)} to ${testEda.postdate_range_utc.max.slice(0, 10)}`;
  document.querySelector("#test-input-caption").textContent =
    `All ${testEda.rows.toLocaleString()} unlabeled test inputs: ` +
    `${testEda.category_frequencies.length} categories; median title ${testEda.title_character_lengths.median} characters; ` +
    `median ${testEda.tag_token_counts.median} tag tokens. No target values were read.`;
  testEda.category_frequencies.forEach((item) => {
    renderTableRow(document.querySelector("#test-category-summary"), [
      item.category,
      item.count.toLocaleString(),
      `${(100 * item.count / testEda.rows).toFixed(2)}%`,
    ]);
  });
  const predictions = document.querySelector("#test-predictions");
  Object.entries(inference.models).forEach(([name, summary]) => {
    renderTableRow(predictions, [
      modelNames[name],
      summary.rows.toLocaleString(),
      summary.min.toFixed(2),
      summary.p10.toFixed(2),
      summary.median.toFixed(2),
      summary.mean.toFixed(2),
      summary.p90.toFixed(2),
      summary.max.toFixed(2),
      summary.outside_train_label_range_count.toLocaleString(),
    ]);
  });
}

async function loadModelsPage() {
  const [data, evaluation] = await Promise.all([getJson("/api/models"), getJson("/api/metrics")]);
  document.querySelector("#fit-rows").textContent = data.fit_rows.toLocaleString();
  const cards = document.querySelector("#model-cards");
  data.models.forEach((model) => {
    const article = document.createElement("article");
    article.className = "model-item";
    const link = document.createElement("a");
    link.href = `/model/${model.name}`;
    link.className = "model-item-link";
    const title = document.createElement("h3");
    title.textContent = model.display_name;
    const description = document.createElement("p");
    description.textContent = model.why;
    const scores = document.createElement("div");
    scores.className = "model-meta";
    scores.textContent = `Chronological SRC ${model.chronological.holdout.src.toFixed(3)} | MAE ${model.chronological.holdout.mae.toFixed(3)}`;
    link.append(title, description, scores);
    article.append(link);
    cards.append(article);
  });
  renderEvaluation(evaluation);
}

function renderMetricBars(root, metrics) {
  Object.entries(metrics).filter(([name]) => name !== "median").forEach(([name, values]) => {
    const article = document.createElement("article");
    article.className = "metric-model";
    const heading = document.createElement("h3");
    heading.textContent = modelNames[name] || name;
    article.append(heading);
    [["SRC", values.holdout.src, values.holdout.src ?? 0], ["MAE", values.holdout.mae, 1 - values.holdout.mae / 3]].forEach(([label, value, width]) => {
      const row = document.createElement("div");
      row.className = "metric-row";
      const text = document.createElement("span");
      text.textContent = label;
      const score = document.createElement("b");
      score.textContent = value == null ? "n/a" : value.toFixed(3);
      const track = document.createElement("div");
      track.className = "metric-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(0, Math.min(1, width)) * 100}%`;
      track.append(fill);
      row.append(text, score);
      article.append(row, track);
    });
    root.append(article);
  });
}

function renderEvaluation(data) {
  const evaluation = data.evaluation;
  document.querySelector("#chronological-count").textContent = `${evaluation.train_rows.toLocaleString()} train / ${evaluation.holdout_rows.toLocaleString()} holdout rows`;
  document.querySelector("#diagnostic-count").textContent = `Full-train chronological holdout: ${evaluation.holdout_rows.toLocaleString()} rows`;
  renderMetricBars(document.querySelector("#chronological-bars"), data.metrics);
  Object.entries(data.metrics).forEach(([name, values]) => {
    renderTableRow(document.querySelector("#chronological-table"), [
      modelNames[name] || "Median baseline", evaluation.train_rows.toLocaleString(), evaluation.holdout_rows.toLocaleString(),
      values.train.src?.toFixed(3), values.train.mae.toFixed(3), values.holdout.src?.toFixed(3),
      values.holdout.mae.toFixed(3), values.holdout.rmse.toFixed(3), values.holdout.r2?.toFixed(3),
    ]);
  });
  if (data.user_aware) {
    const split = data.user_aware.evaluation;
    document.querySelector("#user-aware-results").hidden = false;
    document.querySelector("#user-aware-count").textContent = `${split.train_rows.toLocaleString()} train / ${split.holdout_rows.toLocaleString()} unseen-user holdout posts; ${split.holdout_user_count.toLocaleString()} held-out users; ${split.user_overlap_count} overlap`;
    renderMetricBars(document.querySelector("#user-aware-bars"), data.user_aware.metrics);
  }
  if (data.combined_evaluation) {
    const combined = data.combined_evaluation;
    const split = combined.evaluation;
    document.querySelector("#combined-results").hidden = false;
    document.querySelector("#combined-count").textContent =
      `${split.train_rows.toLocaleString()} train / ${split.holdout_rows.toLocaleString()} future-time unseen-user posts; ` +
      `${split.holdout_user_count.toLocaleString()} held-out users; ${split.user_overlap_count} overlap; ` +
      `${split.purged_rows.toLocaleString()} pre-cutoff rows purged; cutoff ${split.cutoff_utc.slice(0, 10)}.`;
    renderMetricBars(document.querySelector("#combined-bars"), combined.metrics);
    Object.entries(combined.metrics).forEach(([name, values]) => {
      renderTableRow(document.querySelector("#combined-table"), [
        modelNames[name] || "Median baseline",
        values.train.src?.toFixed(3),
        values.train.mae.toFixed(3),
        values.holdout.src?.toFixed(3),
        values.holdout.mae.toFixed(3),
        values.holdout.rmse.toFixed(3),
        values.holdout.r2?.toFixed(3),
      ]);
    });
  }
  if (data.temporal_sensitivity) {
    document.querySelector("#temporal-sensitivity").hidden = false;
    data.temporal_sensitivity.runs.forEach((run) => {
      const metrics = run.holdout_metrics;
      const names = Object.keys(metrics);
      const srcWinner = names.reduce((best, name) => metrics[name].src != null && (best == null || metrics[name].src > metrics[best].src) ? name : best, null);
      const maeWinner = names.reduce((best, name) => best == null || metrics[name].mae < metrics[best].mae ? name : best, null);
      renderTableRow(document.querySelector("#temporal-runs"), [
        `${Math.round(run.holdout_fraction * 100)}% tail / ${run.holdout_rows.toLocaleString()} posts`,
        run.cutoff_utc.slice(0, 10),
        `${modelNames[srcWinner] || srcWinner} (${metrics[srcWinner].src.toFixed(3)})`,
        `${modelNames[maeWinner] || maeWinner} (${metrics[maeWinner].mae.toFixed(3)})`,
      ]);
    });
  }
}

async function loadModelPage() {
  const modelName = location.pathname.split("/").filter(Boolean).pop();
  const model = await getJson(`/api/model/${encodeURIComponent(modelName)}`);
  document.title = `ViralScope | ${model.display_name}`;
  document.querySelector("#try-this-model").href = `/demo?model=${encodeURIComponent(modelName)}`;
  document.querySelector("#model-title").textContent = model.display_name;
  document.querySelector("#model-family").textContent = model.family;
  document.querySelector("#model-why").textContent = model.why;
  document.querySelector("#model-training").textContent = model.training;
  document.querySelector("#model-fit-rows").textContent = model.fit_rows.toLocaleString();
  document.querySelector("#model-path").textContent = model.final_model_path;
  document.querySelector("#model-explanation").textContent = model.explanation;
  document.querySelector("#feature-list").replaceChildren();
  model.features.forEach((feature) => {
    const item = document.createElement("li");
    item.textContent = feature;
    document.querySelector("#feature-list").append(item);
  });

  const metricBody = document.querySelector("#model-metrics");
  renderTableRow(metricBody, ["Chronological train", model.chronological_evaluation.train_rows.toLocaleString(), model.chronological.train.src?.toFixed(3), model.chronological.train.mae.toFixed(3), model.chronological.train.rmse.toFixed(3), model.chronological.train.r2?.toFixed(3)]);
  renderTableRow(metricBody, ["Chronological holdout", model.chronological_evaluation.holdout_rows.toLocaleString(), model.chronological.holdout.src?.toFixed(3), model.chronological.holdout.mae.toFixed(3), model.chronological.holdout.rmse.toFixed(3), model.chronological.holdout.r2?.toFixed(3)]);
  renderTableRow(metricBody, ["Unseen-user train", model.user_disjoint_evaluation.train_rows.toLocaleString(), model.user_disjoint.train.src?.toFixed(3), model.user_disjoint.train.mae.toFixed(3), model.user_disjoint.train.rmse.toFixed(3), model.user_disjoint.train.r2?.toFixed(3)]);
  renderTableRow(metricBody, ["Unseen-user holdout", model.user_disjoint_evaluation.holdout_rows.toLocaleString(), model.user_disjoint.holdout.src?.toFixed(3), model.user_disjoint.holdout.mae.toFixed(3), model.user_disjoint.holdout.rmse.toFixed(3), model.user_disjoint.holdout.r2?.toFixed(3)]);

  const test = model.test_prediction_summary;
  document.querySelector("#test-summary").textContent =
    `${test.rows.toLocaleString()} unlabeled test inputs; mean prediction ${test.mean.toFixed(2)}, ` +
    `median ${test.median.toFixed(2)}, range ${test.min.toFixed(2)} - ${test.max.toFixed(2)}. ` +
    `${test.outside_train_label_range_count.toLocaleString()} predictions fall outside the train label range. No test score is computed.`;
  document.querySelector("#model-limits").textContent =
    "Accuracy and F1 are classification metrics and are not applicable to this continuous regression target. Test labels are withheld, so test SRC, MAE, RMSE, and R2 are unavailable.";
}

async function loadDemoModels() {
  const data = await getJson("/api/models");
  const select = document.querySelector("#demo-model");
  data.models.forEach((model) => {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.display_name;
    select.append(option);
  });
}

function initializeDemo() {
  const form = document.querySelector("#demo-form");
  if (!form) return;
  const demoExamples = [
    {title: "A quiet path through the hills", tags: "hiking trail hills outdoors nature", category: "Travel&Active&Sports", subcategory: "Travel", concept: "landscape", planned_at: "2015-08-22T09:00"},
    {title: "Fresh berry tart from the kitchen", tags: "baking berries dessert homemade food", category: "Food", subcategory: "Dessert", concept: "berries", planned_at: "2015-11-07T14:00"},
    {title: "A bright little bird in the garden", tags: "bird garden wildlife morning nature", category: "Animal", subcategory: "Birds", concept: "garden", planned_at: "2015-06-03T07:30"},
    {title: "A small stage with a big sound", tags: "music live concert stage guitar", category: "Entertainment", subcategory: "Music", concept: "concert", planned_at: "2015-12-05T20:00"}
  ];
  const randomize = () => {
    const sample = demoExamples[Math.floor(Math.random() * demoExamples.length)];
    ["title", "tags", "category", "subcategory", "concept", "planned_at"].forEach((key) => { form.elements[key].value = sample[key]; });
    document.querySelector("#demo-result").textContent = "";
    document.querySelector("#demo-explanations").replaceChildren();
  };
  document.querySelector("#demo-random").addEventListener("click", randomize);
  randomize();
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const payload = Object.fromEntries(new FormData(form).entries());
    payload.planned_at = `${payload.planned_at}:00Z`;
    try {
      const response = await fetch("/api/predict", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Prediction failed.");
      document.querySelector("#demo-result").textContent = `${result.prediction.toFixed(2)} · ${modelNames[result.model]}`;
      const list = document.querySelector("#demo-explanations");
      list.replaceChildren();
      result.contributions.forEach((item) => {
        const row = document.createElement("li");
        row.textContent = `${item.feature}: ${item.contribution >= 0 ? "+" : ""}${item.contribution.toFixed(3)}`;
        list.append(row);
      });
    } catch (error) {
      document.querySelector("#demo-error").textContent = error.message;
    }
  });
}

const page = document.body.dataset.page;
const loaders = {dataset: loadDatasetPage, models: loadModelsPage, model: loadModelPage};
if (loaders[page]) loaders[page]().catch(showError);
if (page === "demo") {
  loadDemoModels().then(() => {
    const requestedModel = new URLSearchParams(location.search).get("model");
    if (requestedModel && modelNames[requestedModel]) {
      document.querySelector("#demo-model").value = requestedModel;
    }
    initializeDemo();
  }).catch(showError);
}