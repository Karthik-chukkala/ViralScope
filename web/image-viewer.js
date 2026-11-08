function initializeImageViewer() {
  const images = [...document.querySelectorAll("main img")];
  if (!images.length) return;

  const dialog = document.createElement("dialog");
  dialog.className = "image-viewer";
  dialog.setAttribute("aria-label", "Image viewer");
  dialog.innerHTML = `
    <div class="image-viewer-toolbar">
      <p class="image-viewer-caption" id="image-viewer-caption"></p>
      <div class="image-viewer-controls">
        <button type="button" data-zoom="out" aria-label="Zoom out">−</button>
        <button type="button" data-zoom="reset" aria-label="Reset zoom">100%</button>
        <button type="button" data-zoom="in" aria-label="Zoom in">+</button>
        <button type="button" data-close aria-label="Close image viewer">Close</button>
      </div>
    </div>
    <div class="image-viewer-viewport">
      <img class="image-viewer-image" alt="">
    </div>`;
  document.body.append(dialog);

  const viewerImage = dialog.querySelector(".image-viewer-image");
  const caption = dialog.querySelector(".image-viewer-caption");
  const viewport = dialog.querySelector(".image-viewer-viewport");
  const resetButton = dialog.querySelector('[data-zoom="reset"]');
  let zoom = 1;

  const updateZoom = (value) => {
    zoom = Math.max(0.5, Math.min(4, value));
    viewerImage.style.transform = `scale(${zoom})`;
    resetButton.textContent = `${Math.round(zoom * 100)}%`;
  };
  const openImage = (image) => {
    viewerImage.src = image.currentSrc || image.src;
    viewerImage.alt = image.alt || "Enlarged image";
    caption.textContent = image.alt || image.closest("figure")?.querySelector("figcaption")?.textContent || "Image preview";
    viewport.scrollTop = 0;
    viewport.scrollLeft = 0;
    updateZoom(1);
    dialog.showModal();
  };

  images.forEach((image) => {
    image.tabIndex = 0;
    image.setAttribute("role", "button");
    image.setAttribute("aria-haspopup", "dialog");
    image.setAttribute("aria-label", `${image.alt || "Image"} — open full-screen view`);
    image.addEventListener("click", () => openImage(image));
    image.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openImage(image);
      }
    });
  });

  dialog.querySelector('[data-zoom="out"]').addEventListener("click", () => updateZoom(zoom - 0.25));
  resetButton.addEventListener("click", () => updateZoom(1));
  dialog.querySelector('[data-zoom="in"]').addEventListener("click", () => updateZoom(zoom + 0.25));
  dialog.querySelector("[data-close]").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
  dialog.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      dialog.close();
    } else if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      updateZoom(zoom + 0.25);
    } else if (event.key === "-") {
      event.preventDefault();
      updateZoom(zoom - 0.25);
    } else if (event.key === "0") {
      event.preventDefault();
      updateZoom(1);
    }
  });
  dialog.addEventListener("close", () => {
    viewerImage.removeAttribute("src");
    updateZoom(1);
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeImageViewer, {once: true});
} else {
  initializeImageViewer();
}
