const RepoBrowser = (() => {
  const breadcrumbEl = document.getElementById("repo-breadcrumb");
  const branchEl = document.getElementById("repo-branch");
  const fileTableEl = document.getElementById("file-table");
  const contentPathEl = document.getElementById("content-path");
  const contentViewEl = document.getElementById("content-view");

  const FOLDER_ICON =
    '<svg class="row-icon" viewBox="0 0 16 16" width="16" height="16" fill="currentColor">' +
    '<path d="M1.75 2A1.75 1.75 0 0 0 0 3.75v8.5C0 13.216.784 14 1.75 14h12.5A1.75 1.75 0 0 0 16 12.25v-7.5A1.75 1.75 0 0 0 14.25 3H7.5L6.03 1.53A1.75 1.75 0 0 0 4.81 1H1.75Z"/></svg>';
  const FILE_ICON =
    '<svg class="row-icon" viewBox="0 0 16 16" width="16" height="16" fill="currentColor">' +
    '<path d="M2 1.75C2 .784 2.784 0 3.75 0h5.086c.464 0 .909.184 1.237.513l2.914 2.914c.329.328.513.773.513 1.237v8.586A1.75 1.75 0 0 1 11.75 15h-8A1.75 1.75 0 0 1 2 13.25Zm1.75-.25a.25.25 0 0 0-.25.25v11.5c0 .138.112.25.25.25h8a.25.25 0 0 0 .25-.25V6h-2.75A1.75 1.75 0 0 1 9 4.25V1.5Zm6.75.062V4.25c0 .138.112.25.25.25h2.688a.252.252 0 0 0-.011-.013l-2.914-2.914a.272.272 0 0 0-.013-.011Z"/></svg>';

  let repoName = "";
  let allFiles = []; // flat list from /api/repo/tree
  let currentDir = ""; // "" = root

  function childrenOfDir(dir) {
    const prefix = dir ? `${dir}/` : "";
    const dirs = new Map(); // name -> path
    const files = [];

    for (const f of allFiles) {
      if (!f.path.startsWith(prefix)) continue;
      const rest = f.path.slice(prefix.length);
      if (!rest) continue;
      const slashIdx = rest.indexOf("/");
      if (slashIdx === -1) {
        files.push(f);
      } else {
        const name = rest.slice(0, slashIdx);
        dirs.set(name, `${prefix}${name}`);
      }
    }

    const dirEntries = [...dirs.entries()]
      .map(([name, path]) => ({ type: "dir", name, path }))
      .sort((a, b) => a.name.localeCompare(b.name));
    const fileEntries = files
      .map((f) => ({ type: "file", name: f.path.slice(prefix.length), path: f.path }))
      .sort((a, b) => a.name.localeCompare(b.name));

    return [...dirEntries, ...fileEntries];
  }

  function renderBreadcrumb() {
    breadcrumbEl.innerHTML = "";

    const rootLink = document.createElement("a");
    rootLink.href = "#";
    rootLink.textContent = repoName;
    rootLink.addEventListener("click", (e) => {
      e.preventDefault();
      showReadme();
    });
    breadcrumbEl.appendChild(rootLink);

    if (!currentDir) return;

    const segments = currentDir.split("/");
    let acc = "";
    segments.forEach((seg, i) => {
      acc = acc ? `${acc}/${seg}` : seg;
      const sep = document.createElement("span");
      sep.className = "crumb-sep";
      sep.textContent = "/";
      breadcrumbEl.appendChild(sep);

      if (i === segments.length - 1) {
        const current = document.createElement("span");
        current.className = "crumb-current";
        current.textContent = seg;
        breadcrumbEl.appendChild(current);
      } else {
        const path = acc;
        const link = document.createElement("a");
        link.href = "#";
        link.textContent = seg;
        link.addEventListener("click", (e) => {
          e.preventDefault();
          navigateDir(path);
        });
        breadcrumbEl.appendChild(link);
      }
    });
  }

  function renderFileTable() {
    fileTableEl.innerHTML = "";
    for (const entry of childrenOfDir(currentDir)) {
      const row = document.createElement("button");
      row.className = `file-row ${entry.type === "dir" ? "is-dir" : "is-file"}`;
      row.innerHTML = `${entry.type === "dir" ? FOLDER_ICON : FILE_ICON}<span class="row-name">${entry.name}</span>`;
      row.addEventListener("click", () => {
        if (entry.type === "dir") {
          navigateDir(entry.path);
        } else {
          openFile(entry.path);
        }
      });
      fileTableEl.appendChild(row);
    }
  }

  function navigateDir(dir) {
    currentDir = dir;
    renderBreadcrumb();
    renderFileTable();
  }

  async function showReadme() {
    currentDir = "";
    renderBreadcrumb();
    renderFileTable();

    contentPathEl.textContent = "README.org";
    contentViewEl.innerHTML = '<p class="muted">Loading README…</p>';
    try {
      const res = await fetch("/api/repo/readme");
      const data = await res.json();
      contentViewEl.innerHTML = data.html;
    } catch (err) {
      contentViewEl.innerHTML = `<p class="chat-error">Failed to load README: ${err}</p>`;
    }
  }

  async function openFile(path) {
    const dir = path.includes("/") ? path.slice(0, path.lastIndexOf("/")) : "";
    navigateDir(dir);

    contentPathEl.textContent = path;
    contentViewEl.innerHTML = '<p class="muted">Loading…</p>';
    document.getElementById("repo-body").scrollTop = 0;

    try {
      const res = await fetch(`/api/repo/file?path=${encodeURIComponent(path)}`);
      if (!res.ok) {
        contentViewEl.innerHTML = `<p class="chat-error">File not found: ${path}</p>`;
        return;
      }
      const data = await res.json();
      if (data.is_binary) {
        contentViewEl.innerHTML = `
          <p class="muted">This file is large or binary (${data.size.toLocaleString()} bytes) and isn't shown inline.</p>
          <p><a href="${data.raw_url}" target="_blank" rel="noopener">View raw on GitHub</a></p>
        `;
        return;
      }
      contentViewEl.innerHTML = data.highlighted_html;
    } catch (err) {
      contentViewEl.innerHTML = `<p class="chat-error">Failed to load file: ${err}</p>`;
    }
  }

  async function init() {
    const res = await fetch("/api/repo/tree");
    const data = await res.json();
    repoName = data.full_name;
    allFiles = data.files;
    branchEl.textContent = data.default_branch;

    await showReadme();
  }

  return { init, openFile };
})();
