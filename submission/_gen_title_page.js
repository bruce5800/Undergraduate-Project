const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, AlignmentType } = require("docx");

const body = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, size: 23, ...opts })],
  });

const heading = (text) =>
  new Paragraph({
    spacing: { before: 220, after: 100 },
    children: [new TextRun({ text, bold: true, size: 24 })],
  });

const kv = (label, value) =>
  new Paragraph({
    spacing: { after: 120 },
    children: [
      new TextRun({ text: label + ": ", bold: true, size: 23 }),
      new TextRun({ text: value, size: 23 }),
    ],
  });

const doc = new Document({
  styles: { default: { document: { run: { font: "Calibri", size: 23 } } } },
  sections: [
    {
      properties: {
        page: {
          size: { width: 12240, height: 15840 },
          margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
        },
      },
      children: [
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 120 },
          children: [new TextRun({ text: "Title Page", bold: true, size: 30 })],
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { after: 300 },
          children: [new TextRun({
            text: "(Double-anonymized submission — not sent to peer reviewers)",
            italics: true, size: 21 })],
        }),
        heading("Manuscript title"),
        body("Pareto-Dominant Reinforcement Learning for Cloud-Edge LLM Inference Scheduling"),
        heading("Author"),
        kv("Name", "Zhuolun Li"),
        kv("Affiliation",
           "Faculty of Science and Engineering, University of Bristol, Bristol, United Kingdom"),
        kv("ORCID", "0009-0004-1487-3862"),
        heading("Address for correspondence"),
        body("Zhuolun Li, Faculty of Science and Engineering, University of Bristol, Bristol, United Kingdom."),
        kv("E-mail", "nu25406@bristol.ac.uk"),
        kv("Corresponding author", "Zhuolun Li"),
        heading("Acknowledgments"),
        body("None."),
        heading("Funding"),
        body("This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."),
        heading("Competing interests"),
        body("The author declares no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."),
        heading("Data and code availability"),
        body("The simulator, all eleven scheduler implementations, and reproducibility manifests for the 30+ experimental configurations will be released as open source upon acceptance; an anonymized archive can be provided during review upon request."),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Title_Page.docx", buf);
  console.log("written Title_Page.docx", buf.length, "bytes");
});
