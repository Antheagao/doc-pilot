import type { Metadata } from "next";
import ApiKeysManager from "@/components/ApiKeysManager";

export const metadata: Metadata = {
  title: "API keys - doc-pilot",
};

export default function ApiKeysPage() {
  return <ApiKeysManager />;
}
