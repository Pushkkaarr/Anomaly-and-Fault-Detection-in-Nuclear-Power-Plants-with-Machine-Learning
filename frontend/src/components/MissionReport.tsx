"use client";

import React from "react";
import { SimulationMetrics, ReactorState } from "@/types/reactor";
import { SCENARIO_INTEL } from "./ScenarioIntelPanel";

interface MissionReportProps {
  metrics: SimulationMetrics | null;
  scenario: string | null;
  history: ReactorState[];
  isRunning: boolean;
}

// ─── Narrative Generator ───────────────────────────────────────────────────────
function buildNarrative(
  metrics: SimulationMetrics,
  scenario: string | null,
  history: ReactorState[]
): {
  verdict: "success" | "partial" | "failure";
  headline: string;
  story: string[];
  crises: Array<{ label: string; value: string; color: string; icon: string }>;
  from: { fuel_temp: string; power: string; pressure: string };
  to: { fuel_temp: string; power: string; pressure: string };
} {
  const intel = scenario ? SCENARIO_INTEL[scenario] : null;
  const success = metrics.total_reward > 0;
  const partialSuccess = !success && metrics.total_reward > -100;

  const verdict: "success" | "partial" | "failure" = success
    ? "success"
    : partialSuccess
    ? "partial"
    : "failure";

  // Derive peak temperature info
  const peakTemp = metrics.max_fuel_temp;
  const peakExceeded = peakTemp > 1100;
  const peakWarning = peakTemp > 950 && !peakExceeded;

  // Build story paragraphs
  const story: string[] = [];

  if (intel) {
    story.push(
      `The AI was deployed against a "${intel.title}" scenario — ${intel.tagline.toLowerCase()}. ` +
        `The system had ${metrics.episode_steps} timesteps to detect the anomaly and return the reactor to safe operating parameters.`
    );
  }

  if (scenario === "lofa") {
    story.push(
      `During the Loss of Flow Accident, coolant circulation dropped sharply, starving the core of heat removal capacity. ` +
        `The AI responded by inserting control rods to reduce fission reactivity while simultaneously commanding emergency coolant flow increases.`
    );
    if (peakExceeded) {
      story.push(
        `⚠️ Despite intervention, fuel temperature peaked at ${peakTemp.toFixed(3)}K — exceeding the 1100K safety threshold. ` +
          `This indicates the AI did not respond fast enough or aggressively enough in the early phase of the transient.`
      );
    } else {
      story.push(
        `Fuel temperature was successfully capped below the 1100K safety limit, peaking at ${peakTemp.toFixed(3)}K. ` +
          `The AI's rod insertion strategy was timely enough to prevent core damage.`
      );
    }
  } else if (scenario === "power_ramp") {
    story.push(
      `The power ramp disturbance demanded a rapid 20–40% increase in reactor output. ` +
        `The AI managed rod withdrawal rate to follow the demand signal while monitoring fuel temperature and pressure closely.`
    );
    story.push(
      `Peak fuel temperature during the ramp was ${peakTemp.toFixed(3)}K. ` +
        (peakTemp > 1000
          ? "This was an aggressive ramp — the AI had to balance speed of response against thermal safety margins."
          : "The controlled ramp remained within safe thermal limits throughout the episode.")
    );
  } else if (scenario === "rod_stuck") {
    story.push(
      `With the control rod stuck and unresponsive, the AI had to rely entirely on coolant flow modulation ` +
        `to control reactor power. This significantly reduces the degree of control available.`
    );
    story.push(
      `The safety events counter recorded ${metrics.safety_events} threshold violations. ` +
        (metrics.safety_events === 0
          ? "Remarkably, the AI maintained stability entirely through coolant adjustment."
          : "These events reflect the difficulty of single-actuator control under a stuck-rod fault.")
    );
  } else if (scenario === "sensor_noise") {
    story.push(
      `Sensor noise corrupted incoming reactor readings with random spikes and offsets. The AI had to filter signal from noise, ` +
        `avoiding over-correction on false alarms while not missing genuine thermal events.`
    );
    story.push(
      `The AI's ${success ? "conservative" : "aggressive"} response strategy resulted in ${
        success ? "smooth" : "oscillatory"
      } control behavior. Average pressure held at ${metrics.avg_pressure.toFixed(3)} bar.`
    );
  } else if (scenario === "normal") {
    story.push(
      `In normal operation, the AI's task is purely steady-state maintenance — keeping power at 100%, fuel temperature ` +
        `in the 900–960K band, and pressure within 9–12 bar, while the baseline model demonstrates its trained capability.`
    );
  }

  // Reward interpretation
  if (success) {
    story.push(
      `Total mission score: ${metrics.total_reward.toFixed(3)} points. A positive score indicates successful stabilization — the AI earned ` +
        `reward for keeping all parameters within bounds across ${metrics.episode_steps} control steps.`
    );
  } else {
    story.push(
      `Total mission score: ${metrics.total_reward.toFixed(3)} points. A negative score means the AI incurred penalties ` +
        `for parameter violations — either from thermal excursions, pressure breaches, or excessive power swings.`
    );
  }

  // Crisis summary
  const crises: Array<{ label: string; value: string; color: string; icon: string }> = [
    {
      label: "Peak Fuel Temp",
      value: `${peakTemp.toFixed(3)} K`,
      color: peakExceeded ? "#ff3b3b" : peakWarning ? "#fbbf24" : "#00ff88",
      icon: peakExceeded ? "🔥" : peakWarning ? "⚠️" : "✓",
    },
    {
      label: "Max Coolant Temp",
      value: `${metrics.max_coolant_temp.toFixed(3)} K`,
      color: metrics.max_coolant_temp > 320 ? "#fbbf24" : "#9ca3af",
      icon: metrics.max_coolant_temp > 320 ? "⚠️" : "✓",
    },
    {
      label: "Avg Pressure",
      value: `${metrics.avg_pressure.toFixed(3)} bar`,
      color:
        metrics.avg_pressure < 8 || metrics.avg_pressure > 13 ? "#fbbf24" : "#00ff88",
      icon: metrics.avg_pressure < 8 || metrics.avg_pressure > 13 ? "⚠️" : "✓",
    },
    {
      label: "Safety Events",
      value: `${metrics.safety_events}`,
      color: metrics.safety_events === 0 ? "#00ff88" : metrics.safety_events < 5 ? "#fbbf24" : "#ff3b3b",
      icon: metrics.safety_events === 0 ? "✓" : "🚨",
    },
    {
      label: "Control Steps",
      value: `${metrics.episode_steps}`,
      color: "#6b8fa8",
      icon: "📊",
    },
    {
      label: "Mission Score",
      value: metrics.total_reward.toFixed(3),
      color: success ? "#00ff88" : partialSuccess ? "#fbbf24" : "#ff3b3b",
      icon: success ? "🏆" : partialSuccess ? "📈" : "❌",
    },
  ];

  // From → To state
  const firstState = history[0];
  const lastState = history[history.length - 1];

  const from = firstState
    ? {
        fuel_temp: `${firstState.fuel_temp.toFixed(3)} K`,
        power: `${(firstState.power * 100).toFixed(3)}%`,
        pressure: `${firstState.pressure.toFixed(3)} bar`,
      }
    : { fuel_temp: "—", power: "—", pressure: "—" };

  const to = lastState
    ? {
        fuel_temp: `${lastState.fuel_temp.toFixed(3)} K`,
        power: `${(lastState.power * 100).toFixed(3)}%`,
        pressure: `${lastState.pressure.toFixed(3)} bar`,
      }
    : { fuel_temp: "—", power: "—", pressure: "—" };

  const headline =
    verdict === "success"
      ? "✅ Reactor Stabilized — AI Mission Successful"
      : verdict === "partial"
      ? "⚠️ Partial Stability — Marginal Control Achieved"
      : "❌ Stabilization Failed — Safety Limits Breached";

  return { verdict, headline, story, crises, from, to };
}

// ─── Component ────────────────────────────────────────────────────────────────
export const MissionReport: React.FC<MissionReportProps> = ({
  metrics,
  scenario,
  history,
  isRunning,
}) => {
  if (!metrics || metrics.episode_steps === 0) return null;

  const { verdict, headline, story, crises, from, to } = buildNarrative(metrics, scenario, history);

  const verdictColor =
    verdict === "success" ? "#00ff88" : verdict === "partial" ? "#fbbf24" : "#ff3b3b";

  const verdictBg =
    verdict === "success"
      ? "rgba(0,255,136,0.06)"
      : verdict === "partial"
      ? "rgba(251,191,36,0.06)"
      : "rgba(255,59,59,0.06)";

  return (
    <div
      className="rounded-xl overflow-hidden"
      style={{
        background: "rgba(3,10,22,0.95)",
        border: `1px solid ${verdictColor}30`,
        boxShadow: `0 0 30px ${verdictColor}08`,
      }}
    >
      {/* Header */}
      <div
        className="px-4 py-3"
        style={{
          background: verdictBg,
          borderBottom: `1px solid ${verdictColor}20`,
        }}
      >
        <div className="flex items-center justify-between">
          <div>
            <p className="text-xs font-bold uppercase tracking-widest" style={{ color: "rgba(107,143,168,0.5)" }}>
              Post-Mission Analysis Report
            </p>
            <h3 className="text-sm font-bold mt-0.5" style={{ color: verdictColor }}>
              {headline}
            </h3>
          </div>
          {!isRunning && (
            <div
              className="text-xs font-bold px-2 py-1 rounded-lg"
              style={{
                background: `${verdictColor}15`,
                border: `1px solid ${verdictColor}30`,
                color: verdictColor,
              }}
            >
              COMPLETE
            </div>
          )}
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Key Metrics Grid */}
        <div className="grid grid-cols-3 gap-2">
          {crises.map((c, i) => (
            <div
              key={i}
              className="rounded-lg p-2 text-center"
              style={{
                background: "rgba(0,0,0,0.3)",
                border: `1px solid ${c.color}20`,
              }}
            >
              <div className="text-lg leading-none mb-1">{c.icon}</div>
              <p className="font-mono font-bold text-xs" style={{ color: c.color }}>
                {c.value}
              </p>
              <p
                className="text-xs uppercase tracking-wide mt-0.5"
                style={{ color: "rgba(107,143,168,0.5)", fontSize: "0.55rem" }}
              >
                {c.label}
              </p>
            </div>
          ))}
        </div>

        {/* State Journey: From → To */}
        <div
          className="rounded-lg p-3"
          style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.04)" }}
        >
          <p
            className="text-xs font-bold uppercase tracking-widest mb-3"
            style={{ color: "rgba(107,143,168,0.5)" }}
          >
            Reactor State Journey
          </p>
          <div className="space-y-2">
            {(
              [
                { label: "Fuel Temp", from: from.fuel_temp, to: to.fuel_temp },
                { label: "Power Output", from: from.power, to: to.power },
                { label: "Pressure", from: from.pressure, to: to.pressure },
              ] as Array<{ label: string; from: string; to: string }>
            ).map((row, i) => (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="w-20 flex-shrink-0" style={{ color: "rgba(107,143,168,0.6)" }}>
                  {row.label}
                </span>
                <span className="font-mono font-bold flex-1" style={{ color: "#ff8f00" }}>
                  {row.from}
                </span>
                <span style={{ color: "rgba(107,143,168,0.3)" }}>→→</span>
                <span
                  className="font-mono font-bold flex-1 text-right"
                  style={{ color: verdictColor }}
                >
                  {row.to}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Narrative Story */}
        <div>
          <p
            className="text-xs font-bold uppercase tracking-widest mb-2"
            style={{ color: "rgba(107,143,168,0.5)" }}
          >
            What Happened
          </p>
          <div className="space-y-2">
            {story.map((paragraph, i) => (
              <p
                key={i}
                className="text-xs leading-relaxed"
                style={{ color: "rgba(160,216,232,0.75)" }}
              >
                {paragraph}
              </p>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default MissionReport;
