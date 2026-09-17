// MathJax setup for pymdownx.arithmatex (generic mode). Material's instant navigation
// swaps page content without a reload, so typesetting reruns on every page change.
window.MathJax = {
  tex: {
    inlineMath: [["\\(", "\\)"]],
    displayMath: [["\\[", "\\]"]],
    processEscapes: true,
    processEnvironments: true,
  },
  options: {
    ignoreHtmlClass: ".*|",
    processHtmlClass: "arithmatex",
  },
};

if (typeof document$ !== "undefined") {
  document$.subscribe(() => {
    if (window.MathJax && window.MathJax.typesetPromise) {
      MathJax.startup.output.clearCache();
      MathJax.typesetClear();
      MathJax.texReset();
      MathJax.typesetPromise();
    }
  });
}
