/* Blind annotations live only in this tab. No network writes or model labels. */
const labels = new Map();
let rows = [],
  index = 0;
const $ = (selector) => document.querySelector(selector);
function setMessage(message) {
  $("#message").textContent = message;
}
function render() {
  const row = rows[index],
    saved = labels.get(row.review_id);
  $("#heading").textContent = `Review ${row.review_id}`;
  $("#text").textContent = row.text;
  $("#position").textContent = `${index + 1} / ${rows.length}`;
  $("#sentiment-label").value = saved?.sentiment_label ?? "";
  $("#issue-label").value = saved?.issue_label ?? "";
  $("#previous").disabled = index === 0;
  $("#next").disabled = index === rows.length - 1;
  $("#progress").textContent =
    `${labels.size} of ${rows.length} labeled in this tab. Download a backup before leaving.`;
  $("#completion").value = labels.size;
  setMessage("");
}
function reviewerName() {
  const name = $("#reviewer").value.trim();
  if (!name || /^[=+\-@\t\r\n]/.test(name) || /[\r\n]/.test(name)) {
    throw new Error(
      "Enter a reviewer identifier that does not start with a spreadsheet formula character or contain a line break.",
    );
  }
  return name;
}
$("#annotation").onsubmit = (event) => {
  event.preventDefault();
  try {
    const reviewer = reviewerName();
    labels.set(rows[index].review_id, {
      review_id: rows[index].review_id,
      sentiment_label: $("#sentiment-label").value,
      issue_label: $("#issue-label").value,
      reviewer,
      reviewed_at: new Date().toISOString(),
    });
    render();
    setMessage("Label saved in this tab only.");
  } catch (error) {
    setMessage(error.message);
  }
};
$("#previous").onclick = () => {
  if (index > 0) {
    index--;
    render();
  }
};
$("#next").onclick = () => {
  if (index + 1 < rows.length) {
    index++;
    render();
  }
};
$("#unlabeled").onclick = () => {
  for (let offset = 1; offset <= rows.length; offset++) {
    const candidate = (index + offset) % rows.length;
    if (!labels.has(rows[candidate].review_id)) {
      index = candidate;
      render();
      return;
    }
  }
  setMessage(
    "All rows labeled. Download the CSV and run the independent evaluation.",
  );
};
$("#download").onclick = () => {
  try {
    reviewerName();
    if (!labels.size)
      throw new Error("Label at least one review before exporting.");
    const columns = [
      "review_id",
      "sentiment_label",
      "issue_label",
      "reviewer",
      "reviewed_at",
    ];
    const escape = (value) => '"' + String(value).replaceAll('"', '""') + '"';
    const csv =
      [
        columns.join(","),
        ...rows
          .filter((row) => labels.has(row.review_id))
          .map((row) =>
            columns
              .map((key) => escape(labels.get(row.review_id)[key]))
              .join(","),
          ),
      ].join("\r\n") + "\r\n";
    const url = URL.createObjectURL(
      new Blob([csv], { type: "text/csv;charset=utf-8" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "feedbacklens-independent-review.csv";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setMessage(
      `Downloaded ${labels.size} labeled rows. The evaluator requires all 200.`,
    );
  } catch (error) {
    setMessage(error.message);
  }
};
fetch("../artifacts/blind_rows.json")
  .then((response) => {
    if (!response.ok) throw new Error("Blind holdout artifact is unavailable.");
    return response.json();
  })
  .then((data) => {
    rows = data;
    render();
  })
  .catch((error) => setMessage(error.message));
