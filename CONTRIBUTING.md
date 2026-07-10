# Contributing Guide

Thank you for your interest in contributing to AI Furniture Detector! This document provides guidelines for developers.

## Getting Started

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-furniture-detector.git
cd ai-furniture-detector

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
make install-dev

# Install pre-commit hooks
make pre-commit-install
```

### Development Workflow

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** with:
   - Clear commit messages
   - Type hints on all functions
   - Docstrings following Google style
   - Tests for new functionality

3. **Run quality checks**
   ```bash
   make lint      # Check code style
   make format    # Auto-format code
   make test      # Run tests
   ```

4. **Push and create a Pull Request**
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

### Python Style Guide

We follow **PEP 8** with these tools:

- **Black**: Code formatter (line length: 120)
- **Ruff**: Linter (with import sorting)
- **isort**: Import organizer
- **MyPy**: Static type checker

Auto-format everything:
```bash
make format
```

### Naming Conventions

- `snake_case` for functions and variables
- `PascalCase` for classes
- `UPPER_CASE` for constants
- Private members start with `_`

### Type Hints

**Required** for all functions:

```python
def detect(
    model: YOLO,
    image_bgr: np.ndarray,
    confidence: float = 0.25,
    iou: float = 0.70,
) -> DetectionResult:
    """
    Run detection pipeline.
    
    Args:
        model: YOLO model instance.
        image_bgr: Input image in BGR format.
        confidence: Detection threshold (0-1).
        iou: NMS threshold (0-1).
    
    Returns:
        DetectionResult with items and annotated image.
    
    Raises:
        ValueError: If parameters are invalid.
    """
    ...
```

### Docstrings

Use **Google style** docstrings:

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description (one line).
    
    Longer description explaining the function's purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When this error occurs.
        TypeError: When that error occurs.
    """
```

## Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_detector.py -v

# Run with coverage report
make test-cov
```

### Writing Tests

- Place tests in `tests/` directory
- Name test files: `test_*.py`
- Name test classes: `Test*`
- Name test methods: `test_*`

Example:
```python
class TestDetector:
    """Tests for detector module."""
    
    def test_load_model_success(self, tmp_path):
        """Test successful model loading."""
        # Arrange
        model_path = tmp_path / "model.pt"
        
        # Act
        model = load_model(model_path)
        
        # Assert
        assert model is not None
    
    def test_load_model_not_found(self):
        """Test error when model not found."""
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/model.pt"))
```

### Test Coverage

Aim for **80%+** coverage. View reports:
```bash
make test-cov
open htmlcov/index.html
```

## Documentation

### Updating README

Changes to API or features? Update `README.md`:

```markdown
## New Feature

Description of feature.

```bash
$ ai-furniture-detector --new-flag
```

Results in:
- Item 1
- Item 2
```

### API Documentation

FastAPI auto-generates docs from docstrings. Example:

```python
@app.post("/api/detect")
async def detect_image(
    image: Annotated[UploadFile, File(description="Room image (JPEG/PNG)")],
    confidence: Annotated[float, Form()] = 0.25,
) -> JSONResponse:
    """
    Detect furniture in uploaded image.
    
    Upload a room image and receive furniture detections with Amazon product links.
    """
```

Visit `/docs` to see auto-generated API documentation.

### Architecture Documentation

Large changes? Update `ARCHITECTURE.md`:
- New modules
- Data flow changes
- Performance optimizations
- Design decisions

## Common Tasks

### Adding a New Feature

1. **Create new module** in `ai_furniture_detector/`
2. **Write tests** in `tests/test_*.py`
3. **Add exports** to `__init__.py`
4. **Update documentation** (README, ARCHITECTURE)

Example: Adding new filter
```python
# ai_furniture_detector/filters.py
def apply_confidence_filter(items: list[DetectedItem], min_conf: float) -> list[DetectedItem]:
    """Filter items by minimum confidence."""
    return [item for item in items if item.confidence >= min_conf]
```

### Adding a New API Endpoint

```python
# ai_furniture_detector/api.py
@app.get("/api/stats")
def get_stats() -> dict[str, Any]:
    """
    Get system statistics.
    
    Returns:
        Dictionary with system stats.
    """
    ...
```

### Fixing a Bug

1. **Create an issue** describing the bug
2. **Create a branch**: `bugfix/issue-description`
3. **Add test** that reproduces the bug
4. **Fix the code**
5. **Verify test passes**
6. **Create PR** referencing the issue

## Pull Request Process

1. **Update** your branch with `main`
2. **Ensure all checks pass**: `make lint test`
3. **Write clear PR description**:
   - What does it do?
   - Why is it needed?
   - How was it tested?
4. **Link related issues**
5. **Request review** from maintainers
6. **Address review feedback**

### PR Template

```markdown
## Description
Brief description of changes.

## Related Issues
Closes #123

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
- [ ] Added/updated tests
- [ ] All tests pass locally
- [ ] Coverage maintained/improved

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Backwards compatible
```

## Releasing a New Version

1. **Update version** in `pyproject.toml`
2. **Update** `CHANGELOG.md` with changes
3. **Create tag**: `git tag v0.2.0`
4. **Push**: `git push origin v0.2.0`
5. **Build**: `python -m build`
6. **Publish**: `twine upload dist/*`

## Getting Help

- **Questions?** Open a GitHub Discussion
- **Found a bug?** Open an Issue
- **Need help?** Check existing Issues and Discussions
- **Chat?** See CONTRIBUTING.md for Discord/Slack links

## Code of Conduct

Be respectful, inclusive, and professional. We're building a positive community!

---

Thank you for contributing! 🚀
