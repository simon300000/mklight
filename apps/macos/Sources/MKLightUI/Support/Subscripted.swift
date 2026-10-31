import SwiftUI

/// Labels written with `_` subscripts (`S_III astigmatism`, `Σ S_I`,
/// `C_L axial`) set as text with real subscripts: the part of a word after
/// `_` is smaller and lowered. Unicode has no subscript capitals, so the
/// Seidel terms need this.
enum Subscripted {
    /// One label piece: plain text, or a subscript.
    struct Piece: Equatable {
        var text: String
        var isSubscript: Bool
    }

    /// Splits `label` into plain and subscript pieces. A subscript runs
    /// from `_` to the end of its word.
    static func pieces(_ label: String) -> [Piece] {
        var out: [Piece] = []
        func add(_ s: Substring, _ sub: Bool) {
            guard !s.isEmpty else { return }
            if let last = out.last, last.isSubscript == sub {
                out[out.count - 1].text += s
            } else {
                out.append(Piece(text: String(s), isSubscript: sub))
            }
        }
        for (k, word) in label.split(separator: " ", omittingEmptySubsequences: false).enumerated() {
            if k > 0 { add(" ", false) }
            if let u = word.firstIndex(of: "_"), u != word.startIndex {
                add(word[..<u], false)
                add(word[word.index(after: u)...], true)
            } else {
                add(word, false)
            }
        }
        return out
    }

    /// The label as `Text`, subscripts at the caption-2 size and lowered.
    static func text(_ label: String) -> Text {
        pieces(label).reduce(Text(verbatim: "")) { t, p in
            let piece = p.isSubscript ? Text(verbatim: p.text).font(.caption2).baselineOffset(-3) : Text(verbatim: p.text)
            return Text("\(t)\(piece)")
        }
    }

    /// The label for VoiceOver: `S_III` reads as `S III`.
    static func spoken(_ label: String) -> String { label.replacingOccurrences(of: "_", with: " ") }
}
