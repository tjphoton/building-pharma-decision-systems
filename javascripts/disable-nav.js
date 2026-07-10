// Nav items pointing at coming_soon.md are placeholders for unwritten
// chapters: keep the title visible but not actually navigable, including
// via keyboard (pointer-events:none in CSS only blocks mouse clicks).
document.addEventListener("DOMContentLoaded", function () {
  document
    .querySelectorAll('a.md-nav__link[href*="coming_soon"]')
    .forEach(function (link) {
      link.setAttribute("aria-disabled", "true");
      link.setAttribute("tabindex", "-1");
      link.addEventListener("click", function (event) {
        event.preventDefault();
      });
    });
});
