// figure dropdown
const dropdownFigures = document.getElementById("dropdown");
const dropdownListFigures = document.getElementById("selectdropdown");
const selectedTextFigures = dropdownFigures.querySelector(".selected");

// toggle dropdown
dropdownFigures.addEventListener("click", function () {
  dropdownListFigures.classList.toggle("hidden");
});

// swap image on selection
dropdownListFigures.addEventListener("click", function (e) {
  if (e.target.tagName === "LI") {
    const imgFile = e.target.dataset.value;
    selectedTextFigures.textContent = e.target.textContent;
    dropdownListFigures.classList.add("hidden");
    document.getElementById("figure-img").src = imgFile;
  }
});
