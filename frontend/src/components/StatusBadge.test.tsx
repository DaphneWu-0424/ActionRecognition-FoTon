// @vitest-environment jsdom
import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { StatusBadge } from "./StatusBadge";

describe("StatusBadge", () => {
  it("renders a user-facing job status", () => {
    render(<StatusBadge status="running" />);
    expect(screen.getByText("分析中")).toBeInTheDocument();
  });
});
