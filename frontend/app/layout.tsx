import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geist = Geist({
  variable: "--font-geist",
  subsets: ["latin"],
  display: "swap",
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Aegis Core | Nuclear Intelligence Platform",
  description:
    "Operational intelligence platform for reactor simulation, anomaly detection, and AI-guided control decisions.",
  icons: {
    icon: "/reactor-favicon.svg",
    shortcut: "/reactor-favicon.svg",
    apple: "/reactor-favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" style={{ background: "#020812" }}>
      <body
        className={`${geist.variable} ${geistMono.variable} antialiased`}
        style={{ background: "#020812", colorScheme: "dark" }}
      >
        {children}
      </body>
    </html>
  );
}
