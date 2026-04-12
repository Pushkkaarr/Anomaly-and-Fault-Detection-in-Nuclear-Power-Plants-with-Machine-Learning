import Link from "next/link";

export default function Home() {
  const coreSignals = [
    { label: "Thermal Margin", value: "88%", tone: "safe" },
    { label: "Pressure Drift", value: "-0.6%", tone: "neutral" },
    { label: "Anomaly Risk", value: "Low", tone: "safe" },
  ];

  return (
    <main className="landing-shell min-h-screen">
      <div className="landing-grid" aria-hidden="true" />

      <section className="landing-hero-wrap">
        <header className="landing-topbar">
          <div className="flex min-w-0 items-center gap-3">
            <div className="landing-orb" />
            <div className="min-w-0">
              <p className="truncate text-[0.65rem] uppercase tracking-[0.2em] text-cyan-200/75 sm:text-xs sm:tracking-[0.22em]">
                Aegis Core
              </p>
              <p className="truncate text-[0.58rem] uppercase tracking-[0.14em] text-cyan-100/45 sm:text-[0.65rem] sm:tracking-[0.18em]">
                Reactor Intelligence Platform
              </p>
            </div>
          </div>
          <Link href="/dashboard" className="landing-mini-button">
            Enter Control Room
          </Link>
        </header>

        <div className="landing-hero-grid">
          <div className="space-y-6">
            <p className="landing-kicker">AI Supervision for Nuclear Stability</p>
            <h1 className="landing-title">
              Pilot reactor scenarios with a control room designed for rapid decisions.
            </h1>
            <p className="max-w-2xl text-sm leading-relaxed text-slate-300 sm:text-[0.95rem] md:text-base">
              Run real-time simulations, monitor LSTM fault forecasts, and inspect AI controller behavior from one unified interface built for high-signal operations.
            </p>

            <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
              <Link href="/dashboard" className="landing-primary-button">
                Launch Live Dashboard
              </Link>
              <a href="#capabilities" className="landing-secondary-button">
                Explore Capabilities
              </a>
            </div>
          </div>

          <aside className="landing-console-card" aria-label="Live operational snapshot">
            <div className="landing-console-header">
              <p>Mission Console</p>
              <span>Live Sync</span>
            </div>

            <div className="landing-console-primary">
              <p className="landing-console-eyebrow">Current Stack</p>
              <p className="landing-console-value">SAC v2 + LSTM Guard</p>
              <p className="landing-console-caption">Decision loop latency below 120ms during nominal load.</p>
            </div>

            <div className="landing-console-signals">
              {coreSignals.map((item) => (
                <div key={item.label} className="landing-signal-row">
                  <div>
                    <p>{item.label}</p>
                    <p>{item.value}</p>
                  </div>
                  <div className={`landing-signal-meter landing-signal-meter-${item.tone}`} />
                </div>
              ))}
            </div>

            <div className="landing-console-footer">
              <div>
                <p>Telemetry</p>
                <p>WebSocket Stream</p>
              </div>
              <div>
                <p>Scenarios</p>
                <p>Adaptive + Stress Tests</p>
              </div>
            </div>
          </aside>
        </div>
      </section>

      <section id="capabilities" className="relative z-10 mx-auto w-full max-w-6xl px-4 pb-12 sm:px-6 sm:pb-16 md:px-10 md:pb-24">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {[
            {
              title: "Control Feedback",
              text: "See autonomous and manual rod/coolant actions with immediate visual response.",
            },
            {
              title: "Fault Intelligence",
              text: "Track risk levels, predicted reactor states, and recommendation hints as conditions evolve.",
            },
            {
              title: "Live Telemetry",
              text: "Monitor temperature, pressure, power, and rates through focused instrumentation panels.",
            },
          ].map((item) => (
            <article key={item.title} className="landing-feature-card">
              <h2>{item.title}</h2>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}
