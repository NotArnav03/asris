"""
ASRIS — Resume Section Parser
Extracts structured sections from raw resume text (education, experience,
skills, projects, certifications, etc.)
"""

import re
from typing import Optional

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from config import SECTION_HEADERS, get_logger

logger = get_logger("preprocessing.section_parser")


class ResumeSection:
    """Represents a parsed section of a resume."""

    def __init__(self, name: str, content: str, start_line: int, end_line: int):
        self.name = name
        self.content = content.strip()
        self.start_line = start_line
        self.end_line = end_line

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return f"ResumeSection(name='{self.name}', lines={self.start_line}-{self.end_line}, preview='{preview}...')"

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def is_empty(self) -> bool:
        return len(self.content.strip()) == 0


class ParsedResume:
    """Container for all parsed sections of a resume."""

    def __init__(self, raw_text: str, sections: dict[str, ResumeSection]):
        self.raw_text = raw_text
        self.sections = sections

    @property
    def education(self) -> Optional[str]:
        return self.sections.get("education", ResumeSection("education", "", 0, 0)).content or None

    @property
    def experience(self) -> Optional[str]:
        return self.sections.get("experience", ResumeSection("experience", "", 0, 0)).content or None

    @property
    def skills(self) -> Optional[str]:
        return self.sections.get("skills", ResumeSection("skills", "", 0, 0)).content or None

    @property
    def projects(self) -> Optional[str]:
        return self.sections.get("projects", ResumeSection("projects", "", 0, 0)).content or None

    @property
    def certifications(self) -> Optional[str]:
        return self.sections.get("certifications", ResumeSection("certifications", "", 0, 0)).content or None

    @property
    def summary(self) -> Optional[str]:
        return self.sections.get("summary", ResumeSection("summary", "", 0, 0)).content or None

    @property
    def section_names(self) -> list[str]:
        return [name for name, sec in self.sections.items() if not sec.is_empty]

    def get_weighted_text(self, weights: Optional[dict[str, float]] = None) -> str:
        """
        Generate weighted text representation. Sections with higher weights
        have their content repeated to amplify importance in embedding space.

        Default weights emphasize skills and experience.
        """
        if weights is None:
            weights = {
                "skills": 3.0,
                "experience": 2.5,
                "education": 1.5,
                "projects": 2.0,
                "certifications": 1.5,
                "summary": 1.0,
                "achievements": 1.0,
                "other": 0.5,
            }

        parts = []
        for section_name, section in self.sections.items():
            if section.is_empty:
                continue
            weight = weights.get(section_name, weights.get("other", 1.0))
            repetitions = max(1, int(weight))
            for _ in range(repetitions):
                parts.append(section.content)

        return "\n\n".join(parts) if parts else self.raw_text

    def to_dict(self) -> dict:
        return {
            name: {
                "content": sec.content,
                "word_count": sec.word_count,
                "lines": f"{sec.start_line}-{sec.end_line}",
            }
            for name, sec in self.sections.items()
            if not sec.is_empty
        }

    def __repr__(self):
        active = [n for n in self.section_names]
        return f"ParsedResume(sections={active})"


def _detect_section_header(line: str) -> Optional[str]:
    """
    Check if a line is a section header. Returns the section name
    if matched, None otherwise.
    """
    clean = line.strip().lower()
    # Remove common prefixes and formatting
    clean = re.sub(r"^[\s\-\*\#\=\|\.\_\~\>\:\d\.]+", "", clean).strip()
    clean = re.sub(r"[\s\-\*\#\=\|\.\_\~\>\:]+$", "", clean).strip()

    if not clean or len(clean) > 60:
        return None

    for section_name, keywords in SECTION_HEADERS.items():
        for keyword in keywords:
            if keyword in clean:
                return section_name

    return None


def parse_resume(text: str) -> ParsedResume:
    """
    Parse a resume text into structured sections.

    Uses header detection to identify section boundaries, then extracts
    the content between headers.
    """
    if not text or not isinstance(text, str):
        return ParsedResume("", {})

    lines = text.split("\n")
    section_boundaries = []

    for i, line in enumerate(lines):
        detected = _detect_section_header(line)
        if detected is not None:
            section_boundaries.append((i, detected))

    # If no sections detected, return everything as "other"
    if not section_boundaries:
        logger.debug("No section headers detected — treating as flat resume")
        return ParsedResume(text, {
            "other": ResumeSection("other", text, 0, len(lines) - 1)
        })

    sections = {}

    # Content before first section header → "header" / summary
    if section_boundaries[0][0] > 0:
        header_content = "\n".join(lines[:section_boundaries[0][0]])
        if header_content.strip():
            sections["header"] = ResumeSection(
                "header", header_content, 0, section_boundaries[0][0] - 1
            )

    # Extract each section
    for idx, (line_num, section_name) in enumerate(section_boundaries):
        # Find end of this section
        if idx + 1 < len(section_boundaries):
            end_line = section_boundaries[idx + 1][0] - 1
        else:
            end_line = len(lines) - 1

        content = "\n".join(lines[line_num + 1: end_line + 1])

        # If section already exists, merge (handles duplicate headers)
        if section_name in sections:
            existing = sections[section_name]
            merged_content = existing.content + "\n" + content
            sections[section_name] = ResumeSection(
                section_name, merged_content, existing.start_line, end_line
            )
        else:
            sections[section_name] = ResumeSection(
                section_name, content, line_num, end_line
            )

    parsed = ParsedResume(text, sections)
    logger.debug(f"Parsed resume into {len(parsed.section_names)} sections: {parsed.section_names}")
    return parsed


def parse_resume_batch(texts: dict[str, str]) -> dict[str, ParsedResume]:
    """Parse a batch of resumes. Input: {filename: text}."""
    results = {}
    for i, (filename, text) in enumerate(texts.items()):
        results[filename] = parse_resume(text)
        if (i + 1) % 200 == 0:
            logger.info(f"Parsed {i + 1}/{len(texts)} resumes")
    logger.info(f"Parsed {len(results)} resumes total")
    return results


if __name__ == "__main__":
    sample = """
    John Doe
    Senior Software Engineer | San Francisco, CA

    PROFESSIONAL SUMMARY
    Passionate engineer with 7+ years building scalable distributed systems.

    EXPERIENCE
    Senior Software Engineer — Google (2021–Present)
    - Designed microservices handling 50M requests/day
    - Led migration from monolith to event-driven architecture

    Software Engineer — Amazon (2018–2021)
    - Built real-time recommendation engine using collaborative filtering
    - Reduced latency by 40% through Redis caching layer

    EDUCATION
    M.S. Computer Science — Stanford University (2018)
    B.S. Computer Science — UC Berkeley (2016)

    TECHNICAL SKILLS
    Python, Java, Go, Kubernetes, Docker, AWS, GCP, PostgreSQL, Redis,
    TensorFlow, PyTorch, Apache Kafka, GraphQL

    PROJECTS
    Open-Source Search Engine — Built a distributed search engine
    ML Pipeline Framework — End-to-end ML training & deployment

    CERTIFICATIONS
    AWS Solutions Architect — Professional
    Google Cloud Professional Data Engineer
    """

    parsed = parse_resume(sample)
    print(parsed)
    print()
    for name, sec in parsed.sections.items():
        print(f"[{name.upper()}] ({sec.word_count} words)")
        print(f"  {sec.content[:120]}...")
        print()
