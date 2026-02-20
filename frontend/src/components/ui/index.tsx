"use client";

import React from "react";

/**
 * Basic Card component for grouping content
 */
export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => (
  <div
    className={`rounded-lg border border-gray-200 bg-white shadow-sm dark:border-gray-800 dark:bg-gray-950 ${className}`}
    {...props}
  />
);

export const CardHeader: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => (
  <div
    className={`border-b border-gray-200 px-6 py-4 dark:border-gray-800 ${className}`}
    {...props}
  />
);

export const CardTitle: React.FC<React.HTMLAttributes<HTMLHeadingElement>> = ({
  className = "",
  ...props
}) => (
  <h2
    className={`text-xl font-semibold leading-none tracking-tight ${className}`}
    {...props}
  />
);

export const CardContent: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({
  className = "",
  ...props
}) => <div className={`px-6 py-4 ${className}`} {...props} />;

/**
 * Button component with variants
 */
interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "danger" | "ghost";
  size?: "sm" | "md" | "lg";
}

export const Button: React.FC<ButtonProps> = ({
  className = "",
  variant = "primary",
  size = "md",
  disabled,
  ...props
}) => {
  const baseStyles =
    "inline-flex items-center justify-center rounded-md font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50";

  const variants = {
    primary: "bg-blue-600 text-white hover:bg-blue-700 focus-visible:ring-blue-600",
    secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200 focus-visible:ring-gray-400",
    danger: "bg-red-600 text-white hover:bg-red-700 focus-visible:ring-red-600",
    ghost: "text-gray-700 hover:bg-gray-100 focus-visible:ring-gray-400",
  };

  const sizes = {
    sm: "px-3 py-1.5 text-sm",
    md: "px-4 py-2 text-base",
    lg: "px-6 py-3 text-lg",
  };

  return (
    <button
      className={`${baseStyles} ${variants[variant]} ${sizes[size]} ${className}`}
      disabled={disabled}
      {...props}
    />
  );
};

/**
 * Input component
 */
export const Input: React.FC<React.InputHTMLAttributes<HTMLInputElement>> = ({
  className = "",
  ...props
}) => (
  <input
    className={`flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-base placeholder-gray-400 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 dark:border-gray-600 dark:bg-gray-950 ${className}`}
    {...props}
  />
);

/**
 * Select component
 */
interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  items?: Array<{ value: string; label: string }>;
}

export const Select: React.FC<SelectProps> = ({
  className = "",
  items,
  children,
  ...props
}) => (
  <select
    className={`flex h-10 w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-base text-gray-900 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-600 focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
    {...props}
  >
    {items?.map((item) => (
      <option key={item.value} value={item.value}>
        {item.label}
      </option>
    ))}
    {children}
  </select>
);

/**
 * Badge component for status/tags
 */
interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "success" | "warning" | "danger" | "info";
}

export const Badge: React.FC<BadgeProps> = ({
  className = "",
  variant = "default",
  ...props
}) => {
  const variants = {
    default: "bg-gray-100 text-gray-900",
    success: "bg-green-100 text-green-900",
    warning: "bg-yellow-100 text-yellow-900",
    danger: "bg-red-100 text-red-900",
    info: "bg-blue-100 text-blue-900",
  };

  return (
    <div
      className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${variants[variant]} ${className}`}
      {...props}
    />
  );
};

/**
 * Loading spinner
 */
export const Spinner: React.FC<{ size?: "sm" | "md" | "lg" }> = ({
  size = "md",
}) => {
  const sizeMap = { sm: "w-4 h-4", md: "w-8 h-8", lg: "w-12 h-12" };
  return (
    <div className={`${sizeMap[size]} animate-spin rounded-full border-4 border-gray-300 border-t-blue-600`} />
  );
};

/**
 * Alert component
 */
interface AlertProps extends React.HTMLAttributes<HTMLDivElement> {
  type?: "info" | "success" | "warning" | "error";
  title?: string;
  onClose?: () => void;
}

export const Alert: React.FC<AlertProps> = ({
  className = "",
  type = "info",
  title,
  onClose,
  children,
  ...props
}) => {
  const bgColors = {
    info: "bg-blue-50 text-blue-900 border-blue-200",
    success: "bg-green-50 text-green-900 border-green-200",
    warning: "bg-yellow-50 text-yellow-900 border-yellow-200",
    error: "bg-red-50 text-red-900 border-red-200",
  };

  return (
    <div
      className={`rounded-lg border px-4 py-3 ${bgColors[type]} ${className}`}
      {...props}
    >
      {title && <h3 className="font-semibold">{title}</h3>}
      {children}
      {onClose && (
        <button
          onClick={onClose}
          className="float-right text-lg font-bold opacity-70 hover:opacity-100"
        >
          ×
        </button>
      )}
    </div>
  );
};
