import { useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import "./style.css";

type World = {
  prompt: string;
  seed: number;
  has_image: boolean;
  paused: boolean;
  world_id: number;
  completed_chunks: number;
  complete: boolean;
  last_chunk_seconds: number | null;
};

function Workspace() {
  const status = useReactor((s) => s.status);
  const connect = useReactor((s) => s.connect);
  const disconnect = useReactor((s) => s.disconnect);
  const send = useReactor((s) => s.sendCommand);
  const upload = useReactor((s) => s.uploadFile);
  const lastError = useReactor((s) => s.lastError);
  const [prompt, setPrompt] = useState("");
  const [seed, setSeed] = useState(0);
  const [file, setFile] = useState<File | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const [world, setWorld] = useState<World | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  useReactorMessage((message) => {
    if (message.type === "state_update") setWorld(message.data as World);
  });
  async function perform(action: () => Promise<unknown>) {
    setError("");
    setBusy(true);
    try {
      await action();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  async function start() {
    if (file) {
      const image = await upload(file);
      await send("set_image", { image, prompt, seed });
    } else await send("start", { prompt, seed });
  }
  const ready = status === "ready";
  return (
    <main>
      <header>
        <strong>
          SGF+ <span>on Reactor</span>
        </strong>
        <span>{status}</span>
        <button
          disabled={busy || status === "connecting" || status === "waiting"}
          onClick={() =>
            void perform(async () => {
              if (ready) {
                await disconnect();
                setWorld(null);
              } else await connect();
            })
          }
        >
          {ready ? "Disconnect" : "Connect"}
        </button>
      </header>
      <div className="workspace">
        <aside>
          <h1>Create a continuous video</h1>
          <p>
            Enter a prompt, optionally choose an image, then press Start.
            Selecting a file alone does not generate.
          </p>
          <label>
            Prompt
            <textarea
              aria-label="Prompt"
              rows={7}
              maxLength={4096}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Describe a scene and its motion…"
            />
          </label>
          <label>
            Reference image (optional)
            <input
              ref={fileInput}
              type="file"
              accept="image/png,image/jpeg,image/webp,image/bmp"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          {file && (
            <button
              onClick={() => {
                setFile(null);
                if (fileInput.current) fileInput.current.value = "";
              }}
            >
              Use text only
            </button>
          )}
          <label>
            Seed
            <input
              type="number"
              min={0}
              max={2147483647}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
          </label>
          <button
            className="primary"
            disabled={!ready || busy || !prompt.trim()}
            onClick={() => void perform(start)}
          >
            {busy ? "Working…" : "Start new video"}
          </button>
          <button
            disabled={!ready || busy || !world?.prompt || world.complete || !prompt.trim()}
            onClick={() => void perform(() => send("set_prompt", { prompt }))}
          >
            Update prompt
          </button>
          <p>
            Update the next generated chunk while keeping the current video.
            Buffered frames keep playing. If paused, the change applies on Resume.
            Use Start new video to begin again.
          </p>
          <button
            disabled={!ready || busy || !world?.prompt || world.complete}
            onClick={() =>
              void perform(() => send("set_paused", { paused: !world?.paused }))
            }
          >
            {world?.paused ? "Resume" : "Pause"}
          </button>
          <p role="status">
            {world
              ? `World ${world.world_id} · ${world.completed_chunks} chunks${world.last_chunk_seconds == null ? "" : ` · ${world.last_chunk_seconds.toFixed(2)} s / chunk`}${world.complete ? " · complete" : ""}`
              : "Connect to begin"}
          </p>
          {(error || lastError) && (
            <p role="alert">{error || String(lastError)}</p>
          )}
        </aside>
        <section className="stage">
          <ReactorView
            track="main_video"
            muted
            videoObjectFit="contain"
            className="video"
          />
          {!world?.completed_chunks && (
            <div className="placeholder">
              {ready
                ? world?.prompt
                  ? "Generating the first frames…"
                  : "Your video will appear here"
                : "Click Connect to open a session"}
            </div>
          )}
        </section>
      </div>
    </main>
  );
}

function App() {
  const [endpoint, setEndpoint] = useState("http://localhost:8080");
  const [active, setActive] = useState(endpoint);
  return (
    <>
      <form
        className="endpoint"
        onSubmit={(e) => {
          e.preventDefault();
          setActive(endpoint);
        }}
      >
        <label>
          Runtime URL{" "}
          <input
            type="url"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
          />
        </label>
        <button>Use endpoint</button>
      </form>
      <ReactorProvider
        key={active}
        modelName="sgf-plus"
        modelTracks={[
          { name: "main_video", kind: "video", direction: "recvonly" },
        ]}
        local
        apiUrl={active}
        connectOptions={{ autoConnect: false }}
      >
        <Workspace />
      </ReactorProvider>
    </>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
