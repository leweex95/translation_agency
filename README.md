# AI Translation Agency

An automated document translation pipeline with AI-powered validation, n8n workflow integration, and self-hosted GitHub Actions support for secure local file processing.

## CLI Commands

### Basic Translation
```bash
# Translate a document
poetry run python -m src.translation_agency.main document.pdf --style professional --target-language hungarian

# Translate with specific style
poetry run python -m src.translation_agency.main document.pdf --style academic --target-language english
```

### Validation Control
```bash
# Skip specific validation steps
poetry run python -m src.translation_agency.main document.pdf --style professional --target-language hungarian --disable-steps grammar style

# Run only translation (skip all validation)
poetry run python -m src.translation_agency.main document.pdf --style professional --target-language hungarian --disable-steps grammar style accuracy hallucination consistency crossllm
```

### Debug Mode
```bash
# Run with visible browser for debugging
poetry run python -m src.translation_agency.main document.pdf --style professional --target-language hungarian --debug --headless false

# Run in headless mode (default)
poetry run python -m src.translation_agency.main document.pdf --style professional --target-language hungarian --headless true
```

### Help
```bash
# Show help
poetry run python -m src.translation_agency.main --help
```

## Local File Translation

With self-hosted runners, translate any local file directly:

1. **Go to GitHub Actions** → Select "AI translation pipeline (Self-hosted)"
2. **Enter your local file path** (e.g., `C:\Users\csibi\Desktop\document.pdf`)
3. **Configure options** (style, language, validation steps)
4. **Run workflow** - processes directly on your machine

The self-hosted runner accesses your local files directly - no uploading required!

## Features

- Multi-format support (PDF, DOCX, TXT, MD)
- AI-powered translation with OpenAI GPT
- Automated validation (grammar, style, accuracy, hallucination, consistency)
- n8n workflow integration
- Self-hosted runner support for local file processing
- Security-first with automated workflow validation
