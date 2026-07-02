import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "BCT Compliance Q&A",
  description: "Bilingual banking regulation assistant",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body>{children}</body>
    </html>
  );
}
