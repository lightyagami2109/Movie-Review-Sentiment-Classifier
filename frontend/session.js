// Lets the API helper in src.jsx ask the sign-in library for a fresh token
// without importing it. auth.jsx fills this in once the account system is ready.
let provider = async () => null;

export const setTokenProvider = fn => { provider = fn || (async () => null); };
export const getAuthToken = () => provider();
