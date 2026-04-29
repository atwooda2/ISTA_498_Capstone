const dropdown = document.getElementById("dropdown");
const dropdownList = document.getElementById("selectdropdown");
const selectedText = dropdown.querySelector(".selected");

//store loaded data
let summaryData = null;
let tsneData = null;

// load all three datasets for plotly/ show tsne first
fetch("tsne_plot_rotate.json")
  .then((response) => response.json())
  .then((fig) => {
    tsneData = fig;
    showTsne();
  });

fetch("prediction_summary.json")
  .then((response) => response.json())
  .then((data) => {
    summaryData = data;
  });

fetch("prediction_statistics.json")
  .then((response) => response.json())
  .then((data) => {
    statisticsData = data;
  });

// 3d tsne chart
function showTsne() {
  tsneData.layout.paper_bgcolor = "#0f172a";
  tsneData.layout.plot_bgcolor = "#1e293b";
  tsneData.layout.font = { color: "#f1f5f9" };
  tsneData.layout.autosize = true;
  delete tsneData.layout.width;
  delete tsneData.layout.height;

  Plotly.purge("plot");
  Plotly.newPlot("plot", tsneData.data, tsneData.layout, { responsive: true });
}

// prediction summary as a bar chart
function showSummary() {
  const stat = "score";
  const categories = Object.keys(summaryData[stat]);
  const values = Object.values(summaryData[stat]);

  const trace = {
    x: categories,
    y: values,
    type: "bar",
    marker: { color: ["#22c55e", "#f59e0b", "#ef4444", "#38bdf8"] },
  };

  //gets rid of old plot so dosnt stack on each other
  Plotly.purge("plot");
  Plotly.newPlot(
    "plot",
    [trace],
    {
      paper_bgcolor: "#0f172a",
      plot_bgcolor: "#1e293b",
      font: { color: "#f1f5f9" },
      title: "Score by Prediction Type",
      autosize: true,
    },
    { responsive: true }
  );
}

// prediction summary as a table
function showTable() {
  const stats = Object.keys(summaryData);

  //need to add html so updates with the JSON here
  let html = `<table class="prediction-table">
    <thead>
      <tr>
        <th></th>
        <th>True Positive</th>
        <th>False Negative</th>
        <th>False Positive</th>
        <th>True Negative</th>
      </tr>
    </thead>
    <tbody>`;

  stats.forEach((stat) => {
    html += `<tr>
      <td style="font-weight: bold; text-align: left;">${stat}</td>
      <td>${summaryData[stat]["True Positive"]}</td>
      <td>${summaryData[stat]["False Negative"]}</td>
      <td>${summaryData[stat]["False Positive"]}</td>
      <td>${summaryData[stat]["True Negative"]}</td>
    </tr>`;
  });

  html += `</tbody></table>`;

  Plotly.purge("plot");
  document.getElementById("plot").innerHTML = html;
}

// dropdown toggle for picking
dropdown.addEventListener("click", function () {
  dropdownList.classList.toggle("hidden");
});

// dropdown selection for visual
dropdownList.addEventListener("click", function (e) {
  if (e.target.tagName === "LI") {
    const value = e.target.dataset.value;
    selectedText.textContent = e.target.textContent;
    dropdownList.classList.add("hidden");

    if (value === "tsne") showTsne();
    else if (value === "summary") showSummary();
    else if (value === "scatter") showTable();
  }
});
