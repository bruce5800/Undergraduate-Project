const fs = require("fs");
const { Document, Packer, Paragraph, TextRun, AlignmentType, HeadingLevel } = require("docx");

const body = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 160 },
    children: [new TextRun({ text, size: 23, ...opts })],
  });

const heading = (text) =>
  new Paragraph({
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 24 })],
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
          spacing: { after: 240 },
          children: [
            new TextRun({ text: "Declaration of Interest Statement", bold: true, size: 30 }),
          ],
        }),
        new Paragraph({
          spacing: { after: 220 },
          children: [
            new TextRun({ text: "Manuscript title: ", bold: true, size: 23 }),
            new TextRun({
              text: "“Pareto-Dominant Reinforcement Learning for Cloud-Edge LLM Inference Scheduling”",
              italics: true,
              size: 23,
            }),
          ],
        }),
        heading("Declaration of interests"),
        body(
          "☒ The author declares that there are no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper."
        ),
        heading("Funding"),
        body(
          "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors."
        ),
        new Paragraph({ spacing: { before: 360 }, children: [new TextRun({ text: "Author: Zhuolun Li", size: 23 })] }),
        new Paragraph({ spacing: { after: 0 }, children: [new TextRun({ text: "Affiliation: Faculty of Science and Engineering, University of Bristol, Bristol, United Kingdom", size: 23 })] }),
        new Paragraph({ spacing: { before: 60 }, children: [new TextRun({ text: "Date: ____________________", size: 23 })] }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync("Declaration_of_Interest.docx", buf);
  console.log("written Declaration_of_Interest.docx", buf.length, "bytes");
});
