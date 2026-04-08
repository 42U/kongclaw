/**
 * Typed model ID for Anthropic models used with pi-ai's getModel().
 * Extracted from getModel's type signature to avoid subpath imports.
 */
import type { getModel } from "@mariozechner/pi-ai";

// Extract the second parameter type when first is "anthropic"
export type AnthropicModelId = Parameters<typeof getModel<"anthropic", any>>[1];
