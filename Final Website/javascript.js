const dropdown = document.getElementById("dropdown");
const dropdownList = document.getElementById("selectdropdown");
const selectedText = dropdown.querySelector(".selected");

let chartData = null;

// load the json
fetch("rank_averages.json")
  .then((response) => response.json())
  .then((data) => {
    chartData = data;

    // show first chart
    makeChart("avg speed", "Average Speed");
  });
``;

function makeChart(stat, label) {
  const groups = chartData.map((d) => d.group);
  const values = chartData.map((d) => d[stat]);

  let trace;

  //checks stat selected and builds the matching chart type
  if (stat === "percentage behind ball") {
    trace = {
      x: values,
      y: groups,
      type: "bar",
      orientation: "h",
      marker: { color: ["#0c4a6e", "#38bdf8", "#f1f5f9"] },
    };

    //creates scatter plot
  } else if (stat === "time high in air") {
    trace = {
      x: groups,
      y: values,
      type: "scatter",
      line: { color: "#38bdf8", width: 3 },
      marker: { color: "#f1f5f9", size: 10 },
    };

    //creates bar graph
  } else {
    trace = {
      x: groups,
      y: values,
      type: "bar",
      marker: { color: ["#0c4a6e", "#38bdf8", "#f1f5f9"] },
    };
  }

  //purges plot before new one so no stacking
  Plotly.purge("plot");
  Plotly.newPlot(
    "plot",
    [trace],
    {
      paper_bgcolor: "#0f172a",
      plot_bgcolor: "#1e293b",
      font: { color: "#f1f5f9" },
      title: label,
      autosize: true,
    },
    { responsive: true }
  );
}

// dropdown toggle
dropdown.addEventListener("click", function () {
  dropdownList.classList.toggle("hidden");
});

// dropdown selection, gives stat/label for charts
dropdownList.addEventListener("click", function (e) {
  if (e.target.tagName === "LI") {
    const stat = e.target.dataset.value;
    selectedText.textContent = e.target.textContent;
    dropdownList.classList.add("hidden");
    makeChart(stat, e.target.textContent);
    document.getElementById("description").textContent = descriptions[stat];
  }
});

//descriptions for tool tip
const descriptions = {
  "avg speed":
    "How fst a player moves around the field. Higher skill players move more efficiently.",
  "amount used while supersonic":
    "How much boost a player uses at top speed. Experts manage boost better.",
  "time high in air":
    "How long a player stays airborne. Aerials take hundreds of hours to master.",
  "percentage on ground":
    "How much time spent on the ground. Beginners stay grounded, experts spend more time in the air.",
  "percentage offensive third":
    "How often a player pushes into the opponent's side. Experts play more agggressively.",
  "time supersonic speed":
    "Time spent at maximum speed. Faster play indicates better movement skills.",
  "avg powerslide time":
    "Average length of powerslides. Skilled players use quicker, more precise slides.",
  "percentage behind ball":
    "How often a player stays on the defensive side. Beginners hang back more.",
};
