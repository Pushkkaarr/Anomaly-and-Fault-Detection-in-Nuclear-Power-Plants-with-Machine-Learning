import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetBrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Nuclear Reactor Control System | SAC Agent",
  description:
    "AI-powered nuclear reactor anomaly & fault detection using Soft Actor-Critic reinforcement learning models. Real-time simulation and visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" style={{ background: "#020812" }}>
      <body
        className={`${inter.variable} ${jetBrainsMono.variable} antialiased`}
        style={{ background: "#020812", colorScheme: "dark" }}
      >
        {children}
      </body>
    </html>
  );
}
