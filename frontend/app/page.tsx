import { execSync } from "child_process";
import HomeClient from "./home-client";

function getLastCommitTimestamp() {
  try {
    return execSync("git log -1 --format=%cI", {
      cwd: process.cwd(),
      encoding: "utf8",
      stdio: ["ignore", "pipe", "ignore"],
    }).trim();
  } catch {
    return new Date().toISOString();
  }
}

function formatLastUpdatedLabel(timestamp: string) {
  const date = new Date(timestamp);

  if (Number.isNaN(date.getTime())) {
    return timestamp;
  }

  return new Intl.DateTimeFormat("en-US", {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: "Asia/Kolkata",
  }).format(date);
}

export default function Home() {
  const lastUpdatedLabel = formatLastUpdatedLabel(getLastCommitTimestamp());

  return <HomeClient lastUpdatedLabel={lastUpdatedLabel} />;
}
