# Notes CLI Design Plan

## Overview
This document outlines the design decisions and schema for the `notes` CLI application.

## Note Schema
- **id**: Integer, auto-increment starting from 1.
- **title**: String, the title of the note.
- **tags**: List of strings, tags associated with the note.
- **body**: String, the main content of the note.
- **citation**: String (optional), citation in BibTeX format.

## Storage
Each note will be stored as an individual JSON file in the `notes_data/` directory located at the workspace root.

## Commands
- `notes add`: Command to add a new note with the following flags:
  - `--title`: Title of the note.
  - `--body`: Body content of the note.
  - `--tags`: Tags associated with the note (comma-separated).
  - `--citation`: Optional citation in BibTeX format.
- `notes ls`: List all notes with their id and title.
- `notes show <id>`: Print the full contents of one note.
- `notes rm <id>`: Delete a note by id.
- `notes search <query>`: Substring match against title and body, case-insensitive.

