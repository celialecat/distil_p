import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Distil — The model distillation lab",
  description: "Turn frontier intelligence into focused, deployable tools.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
