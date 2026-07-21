# InvoxAI — Local-First Invoice Extraction

Turn invoice and receipt images into structured, searchable data — without your documents ever leaving your machine.

## What it does

Upload an invoice image and InvoxAI extracts vendor, date, amount, tax, and category into clean structured data. It runs OCR and AI extraction locally by default, with an optional cloud backend if you need it.

## How it works

1. **Upload** — drop in an invoice or receipt image
2. **OCR** — text is extracted locally from the image
3. **Extract** — an LLM converts raw text into structured fields
4. **Validate & Repair** — output is checked against a strict schema; if anything's wrong, it's automatically corrected in a feedback loop, no hardcoded fallbacks
5. **Review** — confidence-scored fields shown for quick edits before saving
6. **Save & Search** — stored locally, searchable by vendor, date, or category, exportable to CSV

## Why it's different

- **Two backends, same reliability** — run fully offline with a local model, or switch to a cloud API when you need more power. Either way, extraction goes through the same validation and repair process.
- **Privacy by default** — the local mode never sends your documents over the network.
- **No hardcoded parsing shortcuts** — every field is validated against a strict schema, with automatic correction on failure.

## Setup

Requires Python, and either a local Ollama installation (for offline mode) or a Mistral API key (for cloud mode). Install dependencies, start the app, and choose your backend from the sidebar.

## Status

Built for a hackathon focused on private, reliable document extraction.