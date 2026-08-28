import { useEffect, useMemo, useRef, useState } from "react";
import type { FormEvent, KeyboardEvent } from "react";
import "./index.css";

type Tab = "analyzer" | "report" | "chat";
type Mode = "chat" | "report" | "analyzer";

type NodeUsage = {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
};

type NodeUpdate = {
  type: "node_update";
  node: string;
  summary: string;
  details: Record<string, unknown>;
  node_usage?: NodeUsage;
  timestamp_ms: number;
};

type StreamEvent =
  | NodeUpdate
  | { type: "job_started"; mode: Mode; timestamp_ms: number }
  | { type: "chat_final"; assistant_message: string; chat_history: Array<{ role: "user" | "assistant"; content: string }> }
  | { type: "report_final"; report: Record<string, unknown>; download_url: string | null }
  | { type: "analyzer_final"; output: string; graph_url: string | null }
  | { type: "job_cancelled"; timestamp_ms: number }
  | { type: "job_error"; message: string; timestamp_ms: number }
  | { type: "heartbeat"; timestamp_ms: number }
  | { type: "done"; timestamp_ms: number };

const API_BASE = import.meta.env.VITE_API_BASE ?? "";
const TABS: Array<{ id: Tab; label: string }> = [
  { id: "analyzer", label: "Analyzer" },
  { id: "report", label: "Video Raporu" },
  { id: "chat", label: "Sohbet" },
];
const TOOL_NAMES = [
  "run_abnormal_event_segmenter",
  "analyze_video_with_vlm",
  "save_video_segment",
  "get_video_info",
  "detect_and_track_objects",
  "detect_license_plate_regions",
  "read_license_plate_crops",
  "archive_anomaly_clip",
] as const;
const TOOL_NAME_SET = new Set<string>(TOOL_NAMES);

function randomId(): string {
  if (crypto?.randomUUID) return crypto.randomUUID();
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function parseToolCallNames(toolCalls: unknown): string[] {
  if (typeof toolCalls !== "string" || !toolCalls.trim()) return [];
  const names = new Set<string>();
  for (const part of toolCalls.split("|")) {
    const match = part.trim().match(/^([A-Za-z_][\w]*)\s*\(/);
    if (match && TOOL_NAME_SET.has(match[1])) names.add(match[1]);
  }
  return [...names];
}

function parseToolResultNames(results: unknown): string[] {
  if (!Array.isArray(results)) return [];
  const names = new Set<string>();
  for (const item of results) {
    if (!item || typeof item !== "object") continue;
    const name = "name" in item ? String((item as { name?: unknown }).name ?? "") : "";
    if (TOOL_NAME_SET.has(name)) names.add(name);
  }
  return [...names];
}

function formatTraceEvent(event: NodeUpdate, index: number): string {
  const time = new Date(event.timestamp_ms).toLocaleTimeString();
  const usage = formatNodeUsage(event.node_usage);
  const usageLine = usage ? `\nToken: ${usage}` : "";
  const details = Object.keys(event.details || {}).length ? `\n${JSON.stringify(event.details, null, 2)}` : "";
  return `[${index + 1}] ${event.node} · ${time}${usageLine}\n${event.summary}${details}`;
}

function formatFullTrace(events: NodeUpdate[]): string {
  if (!events.length) return "Canlı süreç boş.";
  return events.map((event, index) => formatTraceEvent(event, index)).join("\n\n");
}

function formatNodeUsage(usage: NodeUsage | undefined): string | null {
  if (!usage || usage.total_tokens <= 0) return null;
  return `${usage.total_tokens.toLocaleString("tr-TR")} token`;
}

function App() {
  const [tab, setTab] = useState<Tab>("analyzer");
  const [sessionId, setSessionId] = useState(() => randomId());
  const [activeVideoUrl, setActiveVideoUrl] = useState<string | null>(null);
  const [activeVideoPath, setActiveVideoPath] = useState<string | null>(null);
  const [chatInput, setChatInput] = useState("");
  const [chatHistory, setChatHistory] = useState<Array<{ role: "user" | "assistant"; content: string }>>([]);
  const [trace, setTrace] = useState<NodeUpdate[]>([]);
  const [statusText, setStatusText] = useState("Hazır");
  const [reportJson, setReportJson] = useState<Record<string, unknown> | null>(null);
  const [reportDownloadUrl, setReportDownloadUrl] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [running, setRunning] = useState(false);
  const [errorText, setErrorText] = useState<string | null>(null);
  const [analyzerOutput, setAnalyzerOutput] = useState("");
  const [analyzerGraphUrl, setAnalyzerGraphUrl] = useState<string | null>(null);
  const operationRef = useRef(false);
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [activeTools, setActiveTools] = useState<Set<string>>(() => new Set());

  const eventSourceRef = useRef<EventSource | null>(null);
  const messagesRef = useRef<HTMLDivElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const apiUrl = useMemo(() => API_BASE.replace(/\/$/, ""), []);

  // A fresh page/tab starts a fresh conversation; never reuse invisible context.
  useEffect(() => () => eventSourceRef.current?.close(), []);

  useEffect(() => {
    messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "smooth" });
  }, [chatHistory, trace, running]);

  async function uploadVideo(file: File): Promise<void> {
    if (operationRef.current || activeVideoPath) return;
    operationRef.current = true;
    setUploading(true);
    setErrorText(null);
    try {
      const sid = sessionId;
      const formData = new FormData();
      formData.set("session_id", sid);
      formData.set("file", file);
      const res = await fetch(`${apiUrl}/api/videos`, { method: "POST", body: formData });
      const payload = await res.json();
      if (!res.ok) throw new Error(payload.detail || "Video yüklenemedi.");
      setReportJson(null);
      setReportDownloadUrl(null);
      setAnalyzerOutput("");
      setAnalyzerGraphUrl(null);
      setTrace([]);
      setActiveVideoPath(payload.video_path);
      setActiveVideoUrl(payload.video_url);
      setStatusText("Video yüklendi.");
      pushTrace("video", `Hedef video yüklendi: ${file.name}`, { video_path: payload.video_path });
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Video yükleme hatası.");
    } finally {
      operationRef.current = false;
      setUploading(false);
    }
  }

  function closeStream(): void {
    eventSourceRef.current?.close();
    eventSourceRef.current = null;
  }

  function pushTrace(node: string, summary: string, details: Record<string, unknown> = {}): void {
    setTrace((prev) => [
      ...prev,
      { type: "node_update", node, summary, details, timestamp_ms: Date.now() },
    ]);
  }

  async function copyText(key: string, text: string): Promise<void> {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedKey(key);
      window.setTimeout(() => setCopiedKey((current) => (current === key ? null : current)), 1500);
    } catch {
      setErrorText("Kopyalanamadı.");
    }
  }

  async function startJob(mode: Mode, message = ""): Promise<void> {
    if (operationRef.current) return;
    setErrorText(null);
    if (!activeVideoPath) {
      setErrorText("Önce video yükleyin.");
      return;
    }
    if (mode === "chat" && !message.trim()) {
      setErrorText("Mesaj boş olamaz.");
      return;
    }
    setTrace([]);
    setActiveTools(new Set());
    setStatusText("İş başlatılıyor…");
    operationRef.current = true;
    setRunning(true);
    setJobId(null);
    try {
      const sid = sessionId;
      if (mode === "chat") {
        setChatHistory((prev) => [...prev, { role: "user", content: message.trim() }]);
        setChatInput("");
      } else if (mode === "report") {
        setReportJson(null);
        setReportDownloadUrl(null);
      } else {
        setAnalyzerOutput("");
        setAnalyzerGraphUrl(null);
      }
      const response = await fetch(`${apiUrl}${mode === "chat" ? "/api/chat" : mode === "report" ? "/api/report" : "/api/jobs/analyzer"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(mode === "chat" ? { session_id: sid, message: message.trim() } : { session_id: sid }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || "İş başlatılamadı.");
      setJobId(payload.job_id);
      setRunning(true);
      setStatusText("Çalışıyor…");
      openStream(payload.job_id, sid);
    } catch (error) {
      operationRef.current = false;
      setRunning(false);
      setStatusText("İş başlatılamadı.");
      throw error;
    }
  }

  function openStream(nextJobId: string, sid: string): void {
    closeStream();
    const source = new EventSource(`${apiUrl}/api/stream/${nextJobId}?session_id=${encodeURIComponent(sid)}`);
    eventSourceRef.current = source;
    source.onopen = () => {
      if (eventSourceRef.current === source) setErrorText(null);
    };
    source.onmessage = (event) => {
      if (eventSourceRef.current === source) handleStreamEvent(JSON.parse(event.data) as StreamEvent);
    };
    source.onerror = () => {
      if (eventSourceRef.current !== source) return;
      // EventSource reconnects; do not mark a still-running server job as idle.
      setStatusText("Akış bağlantısı yeniden kuruluyor; iş sürüyor olabilir.");
      setErrorText("Canlı akış bağlantısı koptu.");
    };
  }

  function handleStreamEvent(event: StreamEvent): void {
    if (event.type === "job_started") {
      setActiveTools(new Set());
      setStatusText(`${event.mode === "chat" ? "Sohbet" : event.mode === "report" ? "Rapor" : "Analyzer"} işi başladı.`);
      return;
    }
    if (event.type === "node_update") {
      setTrace((prev) => [...prev, event]);
      setStatusText(`[${event.node}]`);
      if (event.node === "executor") {
        const started = parseToolCallNames(event.details.tool_calls);
        if (started.length) {
          setActiveTools((prev) => {
            const next = new Set(prev);
            for (const name of started) next.add(name);
            return next;
          });
        }
      } else if (event.node === "tools") {
        const finished = parseToolResultNames(event.details.tool_results);
        if (finished.length) {
          setActiveTools((prev) => {
            const next = new Set(prev);
            for (const name of finished) next.delete(name);
            return next;
          });
        }
      }
      return;
    }
    if (event.type === "chat_final") {
      setChatHistory(event.chat_history);
      setStatusText("Sohbet tamamlandı.");
      return;
    }
    if (event.type === "analyzer_final") {
      setAnalyzerOutput(event.output);
      setAnalyzerGraphUrl(event.graph_url);
      setStatusText("Segmentler hazır.");
      pushTrace("analyzer", "Analiz tamamlandı", { output: event.output, graph_url: event.graph_url });
      return;
    }
    if (event.type === "report_final") {
      setReportJson(event.report);
      setReportDownloadUrl(event.download_url);
      setStatusText("Rapor hazır.");
      return;
    }
    if (event.type === "job_error") {
      setErrorText(event.message);
      setActiveTools(new Set());
      setStatusText("İş hata ile bitti.");
      return;
    }
    if (event.type === "job_cancelled") {
      setActiveTools(new Set());
      setStatusText("İş iptal edildi.");
      return;
    }
    if (event.type === "done") {
      operationRef.current = false;
      setJobId(null);
      setRunning(false);
      setActiveTools(new Set());
      closeStream();
    }
  }

  async function cancelJob(): Promise<void> {
    if (!jobId || !sessionId) return;
    const response = await fetch(`${apiUrl}/api/jobs/${jobId}/cancel`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    if (!response.ok) throw new Error("İptal isteği gönderilemedi.");
    setStatusText("İptal istendi; devam eden model adımının bitmesi bekleniyor.");
  }

  async function newChat(): Promise<void> {
    if (operationRef.current) return;
    operationRef.current = true;
    setUploading(true);
    if (jobId && sessionId) {
      try {
        await fetch(`${apiUrl}/api/jobs/${jobId}/cancel`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId }),
        });
      } catch {
        // ignore cancel errors during full reset
      }
    }
    closeStream();

    const oldSid = sessionId;
    if (oldSid) {
      try {
        await fetch(`${apiUrl}/api/sessions/${oldSid}/clear`, { method: "POST" });
      } catch {
        // ignore; still reset UI
      }
    }

    const nextSid = randomId();
    setSessionId(nextSid);
    try {
      await fetch(`${apiUrl}/api/sessions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: nextSid }),
      });
    } catch {
      // session will be created on next upload/action
    }

    setActiveVideoPath(null);
    setActiveVideoUrl(null);
    setChatInput("");
    setChatHistory([]);
    setTrace([]);
    setActiveTools(new Set());
    setStatusText("Hazır");
    setReportJson(null);
    setReportDownloadUrl(null);
    setJobId(null);
    setUploading(false);
    setRunning(false);
    operationRef.current = false;
    setErrorText(null);
    setAnalyzerOutput("");
    setAnalyzerGraphUrl(null);
    setCopiedKey(null);
    setTab("analyzer");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  function onSubmitChat(event: FormEvent): void {
    event.preventDefault();
    startJob("chat", chatInput).catch((error: unknown) => {
      setErrorText(error instanceof Error ? error.message : "Sohbet başlatılamadı.");
    });
  }

  function onChatKeyDown(event: KeyboardEvent<HTMLTextAreaElement>): void {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      onSubmitChat(event);
    }
  }

  async function runAnalyzer(): Promise<void> {
    try {
      await startJob("analyzer");
    } catch (error) {
      setErrorText(error instanceof Error ? error.message : "Analyzer başlatılamadı.");
    }
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>Neokortex</h1>
        </div>
        <button className="primary full" type="button" disabled={running || uploading} onClick={() => newChat().catch(() => null)}>
          Yeni sohbet
        </button>
        <label className="field">
          <span>Hedef video</span>
          <input
            ref={fileInputRef}
            className="file"
            type="file"
            accept="video/*"
            disabled={uploading || running || !!activeVideoPath}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) uploadVideo(file);
            }}
          />
        </label>
        {activeVideoPath ? <p className="meta">Video değiştirmek için yeni sohbet başlatın. {activeVideoPath.split("/").pop()}</p> : <p className="meta">Video yüklenmedi.</p>}
        {activeVideoUrl ? <video className="side-video" controls src={`${apiUrl}${activeVideoUrl}`} /> : null}
        <div className="tools-panel">
          <h2>Tools</h2>
          <ul className="tools-list">
            {TOOL_NAMES.map((name) => (
              <li key={name} className={activeTools.has(name) ? "running" : undefined}>
                {name}
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <section className="main">
        <header className="topbar">
          <nav className="tabs">
            {TABS.map((item) => (
              <button key={item.id} className={tab === item.id ? "active" : ""} type="button" onClick={() => setTab(item.id)}>
                {item.label}
              </button>
            ))}
          </nav>
          <div className="status">
            <span>{statusText}</span>
            {running ? <span className="pill running">Çalışıyor</span> : <span className="pill">Boşta</span>}
          </div>
        </header>

        <div className="workspace">
          <div className="chat-shell">
            <div className="chat-col">
          {tab === "analyzer" ? (
            <div className="form-page">
              <div className="panel">
                <h2>Analyzer</h2>
                <p className="help">Soldaki hedef videoyu kullanır. clip_size / overlap / fps `.env` (`AS_*`); eşik içeride sabittir.</p>
                <p className="meta">{activeVideoPath ? activeVideoPath.split("/").pop() : "Önce soldan video yükleyin."}</p>
                <div className="row">
                  <button className="primary" type="button" disabled={running || uploading || !activeVideoPath} onClick={() => runAnalyzer()}>
                    çalıştır
                  </button>
                </div>
                <div className="field">
                  <span>segmentler</span>
                  <pre className="output">{analyzerOutput || "Henüz çıktı yok."}</pre>
                </div>
                {analyzerGraphUrl ? <img className="preview" src={`${apiUrl}${analyzerGraphUrl}`} alt="Analyzer grafik" /> : null}
              </div>
            </div>
          ) : null}

          {tab === "report" ? (
            <div className="form-page">
              <div className="panel">
                <h2>Video Raporu</h2>
                <p className="help">Anormal aralıkları analiz eder ve denetlenmiş JSON üretir. Sohbetten bağımsızdır.</p>
                <div className="row">
                  <button
                    className="primary"
                    type="button"
                    disabled={running || uploading}
                    onClick={() => startJob("report").catch((error: unknown) => setErrorText(error instanceof Error ? error.message : "Rapor başlatılamadı."))}
                  >
                    Rapor oluştur
                  </button>
                  <button className="danger" type="button" disabled={!running || !jobId} onClick={() => cancelJob().catch(() => null)}>
                    İptal
                  </button>
                </div>
                {reportJson ? <pre className="output">{JSON.stringify(reportJson, null, 2)}</pre> : <p className="meta">Henüz rapor yok.</p>}
                {reportDownloadUrl ? (
                  <a href={`${apiUrl}${reportDownloadUrl}`} download>
                    JSON indir
                  </a>
                ) : null}
              </div>
            </div>
          ) : null}

          {tab === "chat" ? (
            <>
                <div className="messages" ref={messagesRef}>
                  <div className="messages-inner">
                    {chatHistory.length === 0 && !running ? (
                      <div className="empty">
                        <h2>How can I help you today?</h2>
                        <p>Videoyu yükleyin, sonra anomalileri, zaman aralığını veya kişileri sorun.</p>
                      </div>
                    ) : null}
                    {chatHistory.map((message, index) => (
                      <div key={`${message.role}-${index}`} className={`msg ${message.role}`}>
                        {message.role === "assistant" ? <div className="avatar" /> : null}
                        <div className="bubble">{message.content}</div>
                      </div>
                    ))}
                    {running && tab === "chat" ? (
                      <div className="msg assistant">
                        <div className="avatar pulse" />
                        <div className="loading">
                          <span className="dot" />
                          <span className="dot" />
                          <span className="dot" />
                          <span className="loading-label">
                            {trace.at(-1)?.node ? `${trace.at(-1)?.node} çalışıyor` : "Yanıt hazırlanıyor"}
                          </span>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
                <div className="composer-wrap">
                  <form className="composer" onSubmit={onSubmitChat}>
                    <textarea
                      value={chatInput}
                      placeholder="Mesajınızı yazın..."
                      rows={2}
                      disabled={running || uploading}
                      onChange={(event) => setChatInput(event.target.value)}
                      onKeyDown={onChatKeyDown}
                    />
                    <button className="send" type="submit" disabled={running || uploading} aria-label="Gönder">
                      ➤
                    </button>
                  </form>
                </div>
            </>
          ) : null}
            </div>
              <aside className="trace-col">
                <div className="trace-head">
                  <h2>Canlı süreç</h2>
                  <div className="trace-head-actions">
                    <button
                      className="ghost copy-btn"
                      type="button"
                      disabled={!trace.length}
                      onClick={() => copyText("all", formatFullTrace(trace))}
                    >
                      {copiedKey === "all" ? "Kopyalandı" : "Tümünü kopyala"}
                    </button>
                  </div>
                </div>
                {trace.length === 0 ? <p className="meta">Tüm sekmelerin çalışma adımları burada akar.</p> : null}
                {trace.map((event, index) => {
                  const nodeUsage = formatNodeUsage(event.node_usage);
                  return (
                  <article key={`${event.node}-side-${index}`} className="step">
                    <header>
                      <span className="node">{event.node}</span>
                      <span className="step-actions">
                        {nodeUsage ? <span className="step-usage">{nodeUsage}</span> : null}
                        <span>{new Date(event.timestamp_ms).toLocaleTimeString()}</span>
                        <button
                          className="ghost copy-btn"
                          type="button"
                          onClick={() => copyText(`card-${index}`, formatTraceEvent(event, index))}
                        >
                          {copiedKey === `card-${index}` ? "Kopyalandı" : "Kopyala"}
                        </button>
                      </span>
                    </header>
                    <p>{event.summary}</p>
                    <details>
                      <summary>Detay</summary>
                      <pre>{JSON.stringify(event.details, null, 2)}</pre>
                    </details>
                  </article>
                  );
                })}
                {running ? (
                  <button className="danger" type="button" disabled={!jobId} onClick={() => cancelJob().catch((error: unknown) => setErrorText(String(error)))}>
                    İptal
                  </button>
                ) : null}
              </aside>
          </div>
        </div>
        {errorText ? <p className="error" style={{ padding: "0 1rem 1rem" }}>{errorText}</p> : null}
      </section>
    </div>
  );
}

export default App;
