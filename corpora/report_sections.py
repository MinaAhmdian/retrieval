import re

HEADER = re.compile(r"^[ \t]*([A-Z][A-Z ]{2,40}?)[ \t]*:", re.MULTILINE)


class ReportSections:

    def __init__(self, text: str):
        self.sections: dict[str, list[str]] = {}
        matches = list(HEADER.finditer(text))
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            body = " ".join(text[match.end():end].split())
            self.sections.setdefault(match.group(1), []).append(body)

    def joined(self, headers: list[str]) -> str:
        parts = [body for header in headers for body in self.sections.get(header, []) if body]
        return " ".join(parts)

    def first(self, headers: list[str]) -> str:
        for header in headers:
            text = self.joined([header])
            if text:
                return text
        return ""
