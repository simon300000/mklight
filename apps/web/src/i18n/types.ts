// The shape of a UI string catalog. English (`en/`) is the source: its type
// is the catalog's type, so another language that misses a key (or has one
// English does not) fails `tsc`.

/** A string English writes in two forms; `t()` picks one by `count`. */
export interface Plural {
  one: string
  other: string
}

type Leaf = string | Plural

/** A catalog in another language: the same keys; a plural may be one string
 *  (Chinese has no plural forms). */
export type Localized<T> = {
  [K in keyof T]: T[K] extends Plural ? string | Plural : T[K] extends string ? string : Localized<T[K]>
}

/** Dotted paths of the leaves: `app.menu.file`. */
export type Paths<T, P extends string = ''> = {
  [K in keyof T & string]: T[K] extends Leaf ? `${P}${K}` : Paths<T[K], `${P}${K}.`>
}[keyof T & string]

/** Values for `{name}` placeholders; `count` also picks a plural form. */
export type Params = Record<string, string | number>
