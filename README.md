# AI Translation Agency

## CLI Commands

### Basic Translation
```bash
# Translate a document
poetry run python -m src.translation_agency.main document.txt --style professional --target-language hungarian

# Translate with specific style
poetry run python -m src.translation_agency.main document.txt --style academic --target-language english
```

### Validation Control
```bash
# Skip specific validation steps
poetry run python -m src.translation_agency.main document.txt --style professional --target-language hungarian --disable-steps grammar style

# Run only translation (skip all validation)
poetry run python -m src.translation_agency.main document.txt --style professional --target-language hungarian --disable-steps grammar style accuracy hallucination consistency crossllm
```

### Debug Mode
```bash
# Run with visible browser
poetry run python -m src.translation_agency.main document.txt --style professional --target-language hungarian --debug --headless
```

### Help
```bash
# Show help
poetry run python -m src.translation_agency.main --help
```