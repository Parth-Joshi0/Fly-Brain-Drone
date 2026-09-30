/* FuityFly whitepaper: shared layout.
   To add a page: add it to PAGES below, then copy _template.html. */

const SITE = {
  name: "FuityFly",
  authors: "Akshin Makkar, Parth Joshi, Amal Chalayil Sreekumar",
  year: "2026",
  repo: "https://github.com/Parth-Joshi0/Fly-Brain-Rover",
};

const PAGES = [
  { file: "index.html", label: "Overview", title: "Overview" },
  { file: "banana.html", label: "Banana CNN", title: "Banana recognition network" },
  { file: "brain.html", label: "Fly Brain", title: "Fly-brain neural pathways" },
  { file: "sdk.html", label: "Tello SDK", title: "DJI Tello SDK integration" },
  { file: "sim.html", label: "Simulation", title: "PyBullet simulation" },
];

const current = location.pathname.split("/").pop() || "index.html";
const index = Math.max(0, PAGES.findIndex(p => p.file === current));

function renderNav() {
  const el = document.getElementById("site-nav");
  if (!el) return;
  const links = PAGES.map(p =>
    `<li><a href="${p.file}"${p.file === PAGES[index].file ? ' aria-current="page"' : ""}>${p.label}</a></li>`
  ).join("");
  el.outerHTML = `
    <nav class="nav" aria-label="Primary">
      <div class="nav-inner">
        <a class="brand" href="index.html">${SITE.name}</a>
        <button class="menu-btn" aria-expanded="false">Menu</button>
        <ul class="nav-links">${links}<li><a href="${SITE.repo}">GitHub</a></li></ul>
      </div>
    </nav>`;
  const btn = document.querySelector(".menu-btn");
  const list = document.querySelector(".nav-links");
  btn.addEventListener("click", () => {
    btn.setAttribute("aria-expanded", String(list.classList.toggle("open")));
  });
}

/* Builds the side table of contents from every <section id> with an <h2>. */
function renderToc() {
  const el = document.getElementById("toc");
  if (!el) return;
  const items = [...document.querySelectorAll("main section[id] > h2")].map(h => {
    const text = h.textContent.replace(h.querySelector(".sec")?.textContent || "", "").trim();
    return `<li><a href="#${h.parentElement.id}">${text}</a></li>`;
  });
  el.innerHTML = `<p>On this page</p><ol>${items.join("")}</ol>`;
}

function renderPager() {
  const main = document.querySelector("main");
  if (!main) return;
  const prev = PAGES[index - 1];
  const next = PAGES[index + 1];
  const nav = document.createElement("nav");
  nav.className = "pager";
  nav.innerHTML =
    (prev ? `<a href="${prev.file}"><small>Previous</small>← ${prev.title}</a>` : "") +
    (next ? `<a class="next" href="${next.file}"><small>Next</small>${next.title} →</a>` : "");
  main.appendChild(nav);
}

function renderFooter() {
  const el = document.getElementById("site-footer");
  if (!el) return;
  el.outerHTML = `<footer class="site-footer">${SITE.name} · ${SITE.authors} · ${SITE.year}</footer>`;
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
renderPager();
renderFooter();
renderMath();
