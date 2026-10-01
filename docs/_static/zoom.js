// The tree figures carry small labels that are unreadable at page width, so
// make every figure in the page body a link to the full-resolution image.
document.addEventListener("DOMContentLoaded", () => {
  for (const img of document.querySelectorAll("article img")) {
    if (img.closest("a")) continue;
    const link = document.createElement("a");
    link.href = img.src;
    link.target = "_blank";
    link.rel = "noopener";
    link.className = "figure-zoom";
    img.replaceWith(link);
    link.appendChild(img);
  }
});
