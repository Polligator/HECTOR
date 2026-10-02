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

// Model pages: each shows one of the model drawings in a frame, with a switch
// between the human and the mouse model. The choice is kept in the address
// (#mouse), so a link can open either.
document.addEventListener("DOMContentLoaded", () => {
  for (const view of document.querySelectorAll(".model-view")) {
    const box = view.querySelector(".model-frame-box");
    const frame = view.querySelector("iframe");
    const open = view.querySelector(".model-open");
    const buttons = view.querySelectorAll(".model-switch button");
    const page = view.dataset.page;
    const tall = Number(view.dataset.tall) || 1;
    const extra = Number(view.dataset.extra) || 0;

    // Fit the frame to the page inside it: the bottom of its lowest piece of
    // content. (The page's own scroll height cannot be used, because it never
    // reports less than the frame's current height, so the frame could only
    // ever grow.) A site opened from disk cannot measure its frames, so there
    // the height is estimated from the width.
    const contentHeight = () => {
      try {
        const doc = frame.contentDocument;
        const style = (el) => doc.defaultView.getComputedStyle(el);
        let bottom = 0;
        for (const el of doc.body.children) {
          if (el.matches("script, style")) continue;
          bottom = Math.max(bottom, el.getBoundingClientRect().bottom + (parseFloat(style(el).marginBottom) || 0));
        }
        // Plus the page's own padding below its content.
        return Math.ceil(bottom + (parseFloat(style(doc.body).paddingBottom) || 0));
      } catch (e) {
        return 0;
      }
    };
    const fit = () => {
      frame.style.height = (contentHeight() || Math.round(box.clientWidth * tall + extra)) + "px";
    };
    const show = (model) => {
      const src = `../_static/models/${model}_${page}.html`;
      buttons.forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.model === model)));
      open.href = src;
      if (!frame.src.endsWith(src.slice(2))) frame.src = src;
    };

    buttons.forEach((b) => b.addEventListener("click", () => {
      history.replaceState(null, "", "#" + b.dataset.model);
      show(b.dataset.model);
    }));
    // Measure when the page in the frame has loaded, and once more just after,
    // when its drawing has been fitted to the frame.
    frame.addEventListener("load", () => [0, 200, 1000].forEach((wait) => setTimeout(fit, wait)));
    window.addEventListener("resize", fit);
    show(location.hash === "#mouse" ? "mouse" : "human");
    fit();
  }
});
