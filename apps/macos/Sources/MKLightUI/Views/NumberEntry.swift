import Foundation
import MKLightModel

extension Int {
    /// `v` rounded to the nearest whole number and held to `Int`'s range; 0
    /// for NaN. `Int(v)` traps on a typed `1e19`, on an infinity and on NaN.
    init(saturating v: Double) {
        let r = v.rounded()
        self = Int(exactly: r) ?? (r.isNaN ? 0 : r < 0 ? .min : .max)
    }
}

/// The text of a number field while it has the focus.
enum FieldText {
    /// `v` so that the text reads back as exactly `v`: the compact form
    /// when it does, else every digit. Text left as it was shown then
    /// stands for the stored value, and committing it changes nothing.
    static func exact(_ v: Double) -> String {
        let short = Fmt.short(v)
        return LensValueParser.number(short) == v ? short : NumberDisplay.exact(v)
    }
}
