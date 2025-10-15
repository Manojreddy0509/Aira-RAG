const $ = (id) => document.getElementById(id);

async function uploadFiles() {
  const files = $("files").files;
  const collection = $("collection").value.trim();
  const form = new FormData();
  for (const f of files) form.append("files", f);
  if (collection) form.append("collection", collection);
  const res = await fetch("/upload", { method: "POST", body: form });
  const data = await res.json();
  $("upload-result").textContent = JSON.stringify(data, null, 2);
  $("collection").value = data.collection;
}

async function ask() {
  const collection = $("collection").value.trim();
  const question = $("question").value.trim();
  const k = parseInt($("topk").value || "4", 10);
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ collection, question, k }),
  });
  const data = await res.json();
  $("answer").textContent = data.answer || "";
  const ul = $("sources");
  ul.innerHTML = "";
  (data.sources || []).forEach((s) => {
    const li = document.createElement("li");
    li.textContent = `${s.source || "unknown"} (page ${s.page ?? "?"}) score=${s.score ?? "-"}`;
    ul.appendChild(li);
  });
}

$("btn-upload").addEventListener("click", uploadFiles);
$("btn-ask").addEventListener("click", ask);
