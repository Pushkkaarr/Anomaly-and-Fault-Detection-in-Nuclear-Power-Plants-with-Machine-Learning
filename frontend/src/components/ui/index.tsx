"use client";

import React from "react";

// ─── Card ────────────────────────────────────────────────────
export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => (
  <div
    className={`rounded-lg ${className}`}
    style={{
      background: "rgba(5, 15, 31, 0.85)",
      border: "1px solid rgba(0, 212, 255, 0.12)",
    }}
    {...props}
  />
);

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => (
  <div
    className={`px-4 py-3 ${className}`}
    style={{ borderBottom: "1px solid rgba(0, 212, 255, 0.08)" }}
    {...props}
  />
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  className = "",
  ...props
}) => (
  <h2
    className={`text-sm font-semibold tracking-wide uppercase ${className}`}
    style={{ color: "rgba(0, 212, 255, 0.7)" }}
    {...props}
  />
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => <div className={`px-4 py-3 ${className}`} {...props} />;

// ─── Button ───────────────────────────────────────────────────
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
}

const BUTTON_VARIANTS = {
  primary: {
    background: "rgba(0, 212, 255, 0.15)",
    border: "1px solid rgba(0, 212, 255, 0.4)",
    color: "var(--brand-accent)",
  },
  secondary: {
    background: "rgba(255, 255, 255, 0.05)",
    border: "1px solid rgba(255, 255, 255, 0.12)",
    color: "#a0b8c8",
  },
  danger: {
    background: "rgba(255, 59, 59, 0.12)",
    border: "1px solid rgba(255, 59, 59, 0.4)",
    color: "#ff6568",
  },
  ghost: {
    background: "transparent",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    color: "#6b8fa8",
  },
};

const BUTTON_SIZES = {
  sm: { padding: "4px 12px", fontSize: 12 },
  md: { padding: "8px 16px", fontSize: 14 },
  lg: { padding: "12px 20px", fontSize: 15 },
};

export const Button: React.FC<ButtonProps> = ({
  className = "",
  variant = "primary",
  size = "md",
  disabled,
  style,
  ...props
}) => {
  const v = BUTTON_VARIANTS[variant];
  const s = BUTTON_SIZES[size];
  return (
    <button
      className={`inline-flex items-center justify-center rounded-lg font-semibold transition-all duration-200 ${className}`}
      disabled={disabled}
      style={{
        ...v,
        ...s,
        fontFamily: "Inter, sans-serif",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        ...style,
      }}
      {...props}
    />
  );
};

// ─── Select ──────────────────────────────────────────────────
interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  items?: Array<{ value: string; label: string }>;
}

export const Select: React.FC<SelectProps> = ({
  className = "",
  items,
  children,
  ...props
}) => (
  <select className={`w-full ${className}`} {...props}>
    {items?.map((item) => (
      <option key={item.value} value={item.value}>
        {item.label}
      </option>
    ))}
    {children}
  </select>
);

// ─── Input ────────────────────────────────────────────────────
export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = ({
  className = "",
  style,
  ...props
}) => (
  <input
    className={`w-full rounded-lg px-3 py-2 text-sm outline-none transition-all duration-200 ${className}`}
    style={{
      background: "rgba(5, 15, 31, 0.9)",
      border: "1px solid rgba(0, 212, 255, 0.2)",
      color: "#e2f0ff",
      fontFamily: "Inter, sans-serif",
      ...style,
    }}
    {...props}
  />
);

// ─── Badge ────────────────────────────────────────────────────
interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info";
}

const BADGE_VARIANTS = {
  default: { bg: "rgba(107,143,168,0.15)", color: "#6b8fa8", border: "rgba(107,143,168,0.3)" },
  success: { bg: "rgba(0,255,136,0.12)", color: "var(--brand-accent)", border: "rgba(0,255,136,0.35)" },
  warning: { bg: "rgba(255,214,0,0.1)", color: "#fbbf24", border: "rgba(255,214,0,0.3)" },
  danger: { bg: "rgba(255,59,59,0.12)", color: "#fb2c36", border: "rgba(255,59,59,0.35)" },
  info: { bg: "rgba(0,255,136,0.1)", color: "var(--brand-accent)", border: "rgba(0,255,136,0.3)" },
};

export const Badge: React.FC<BadgeProps> = ({
  className = "",
  variant = "default",
  style,
  ...props
}) => {
  const v = BADGE_VARIANTS[variant];
  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${className}`}
      style={{
        background: v.bg,
        color: v.color,
        border: `1px solid ${v.border}`,
        ...style,
      }}
      {...props}
    />
  );
};

// ─── Spinner ──────────────────────────────────────────────────
export const Spinner: React.FC<{ size?: "sm" | "md" | "lg" }> = ({ size = "md" }) => {
  const sizeMap = { sm: 16, md: 28, lg: 40 };
  const px = sizeMap[size];
  return (
    <div
      className="animate-spin rounded-full"
      style={{
        width: px,
        height: px,
        border: "2px solid rgba(0,255,136,0.15)",
        borderTopColor: "var(--brand-accent)",
      }}
    />
  );
};

// ─── Alert ────────────────────────────────────────────────────
interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  type?: "info" | "success" | "warning" | "error";
  title?: string;
  onClose?: () => void;
}

const ALERT_STYLES = {
  info: { bg: "rgba(0,255,136,0.08)", border: "rgba(0,255,136,0.3)", color: "#a0d8e8" },
  success: { bg: "rgba(0,255,136,0.08)", border: "rgba(0,255,136,0.3)", color: "#aaefcc" },
  warning: { bg: "rgba(255,214,0,0.08)", border: "rgba(255,214,0,0.3)", color: "#ffe680" },
  error: { bg: "rgba(255,59,59,0.1)", border: "rgba(255,59,59,0.35)", color: "#ffaaaa" },
};

export const Alert: React.FC<AlertProps> = ({
  className = "",
  type = "info",
  title,
  onClose,
  children,
  ...props
}) => {
  const s = ALERT_STYLES[type];
  return (
    <div
      className={`rounded-lg px-4 py-3 ${className}`}
      style={{ background: s.bg, border: `1px solid ${s.border}`, color: s.color }}
      {...props}
    >
      <div className="flex items-start justify-between gap-2">
        <div>
          {title && <p className="font-semibold text-sm mb-1">{title}</p>}
          {children}
        </div>
        {onClose && (
          <button
            onClick={onClose}
            className="text-lg font-bold opacity-60 hover:opacity-100 transition-opacity shrink-0"
          >
            ×
          </button>
        )}
      </div>
    </div>
  );
};


