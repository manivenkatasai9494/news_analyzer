async function processUrls() {
  const urls = [
    document.getElementById("url1").value,
    document.getElementById("url2").value
  ].filter(u => u.trim() !== "");

  const res = await fetch("/process-urls", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls })
  });

  const data = await res.json();

  alert(data.status || data.error);
}

async function ask() {
  const question = document.getElementById("question").value;

  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question })
  });

  const data = await res.json();

  document.getElementById("output").innerText =
    data.answer + "\n\nSources:\n" + data.sources.join("\n");
}
