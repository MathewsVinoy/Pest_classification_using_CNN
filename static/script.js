const menuToggle = document.getElementById("menuToggle");
const navLinks = document.getElementById("navLinks");

menuToggle.addEventListener("click", () => {
  navLinks.classList.toggle("show");
});

document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-upload");
  const fileName = document.getElementById("file-name");
  fileInput.addEventListener("change", function () {
    if (fileInput.isDefaultNamespace.length > 0) {
      fileName.textContent = fileInput.files[0].name;
    } else {
      fileName.textContent = "No File Selected";
    }
  });
});
