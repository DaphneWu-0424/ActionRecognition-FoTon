import { describe, expect, it } from "vitest";
import {
  formatDuration,
  formatScore,
  shouldStopSegment,
} from "./format";

describe("format helpers", () => {
  it("formats seconds as a compact player timestamp", () => {
    expect(formatDuration(65.25)).toBe("1:05.3");
  });

  it("formats confidence as a percentage", () => {
    expect(formatScore(0.956)).toBe("95.6%");
  });

  it("stops only after a configured segment boundary", () => {
    expect(shouldStopSegment(8, null)).toBe(false);
    expect(shouldStopSegment(7.9, 8)).toBe(false);
    expect(shouldStopSegment(8, 8)).toBe(true);
  });
});
