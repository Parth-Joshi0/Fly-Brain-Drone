/* FruityFly whitepaper: nav, contents sidebar, footer and math. */

const SITE = {
  name: "FruityFly",
  authors: [
    { name: "Akshin Makkar", linkedin: "https://www.linkedin.com/in/akshin-makkar/" },
    { name: "Parth Joshi", linkedin: "https://www.linkedin.com/in/parth-joshi16/" },
    { name: "Amal Chalayil Sreekumar", linkedin: "https://www.linkedin.com/in/amalchalayilsreekumar/" },
  ],
  year: "2026",
  repo: "https://github.com/Parth-Joshi0/Fly-Brain-Rover",
};

const SECTIONS = [
  { href: "#brain", label: "Brain" },
  { href: "#drone", label: "Drone" },
  { href: "#banana", label: "Banana Model" },
  { href: "#simulation", label: "Simulation" },
];

function renderNav() {
  const el = document.getElementById("site-nav");
  if (!el) return;
  const links = SECTIONS.map(s => `<li><a href="${s.href}">${s.label}</a></li>`).join("");
  el.outerHTML = `
    <nav class="nav" aria-label="Primary">
      <a class="brand" href="#">${SITE.name}</a>
      <ul class="nav-links">${links}<li><a href="${SITE.repo}">GitHub</a></li></ul>
    </nav>`;
}

/* Numbers each section ("1", "1.1") like a LaTeX paper and builds the
   Contents sidebar from them. Abstract and References stay unnumbered. */
function renderToc() {
  const main = document.querySelector("main");
  if (!main) return;
  const items = [];
  let n = 0;
  main.querySelectorAll("section[id]").forEach(sec => {
    const h2 = sec.querySelector("h2");
    if (!h2) return;
    const numbered = sec.id !== "references";
    const tag = numbered ? String(++n) : "";
    if (numbered) h2.insertAdjacentHTML("afterbegin", `<span class="tag">${tag}</span>`);
    const subs = [...sec.querySelectorAll("h3")].map((h3, i) => {
      h3.id ||= `${sec.id}-${i + 1}`;
      const sub = `${tag}.${i + 1}`;
      const text = h3.textContent;
      h3.insertAdjacentHTML("afterbegin", `<span class="tag">${sub}</span>`);
      return `<li class="sub"><a href="#${h3.id}"><span class="tag">${sub}</span>${text}</a></li>`;
    });
    const text = h2.textContent.slice(tag.length);
    items.push(`<li><a href="#${sec.id}"><span class="tag">${tag}</span>${text}</a></li>`, ...subs);
  });
  main.insertAdjacentHTML("beforebegin",
    `<nav class="toc" aria-label="Contents"><p>Contents</p><ol>${items.join("")}</ol></nav>`);
}

function renderFooter() {
  const el = document.getElementById("site-footer");
  if (!el) return;
  const authors = SITE.authors.map(a => `<a href="${a.linkedin}">${a.name}</a>`).join(", ");
  el.outerHTML = `<footer class="site-footer">${SITE.name} · ${authors} · ${SITE.year}</footer>`;
}

function renderMath() {
  if (!window.renderMathInElement) return;
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "\\(", right: "\\)", display: false },
    ],
    throwOnError: false,
  });
}

renderNav();
renderToc();
renderFooter();
renderMath();
