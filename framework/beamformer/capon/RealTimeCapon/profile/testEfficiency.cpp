
#include <Python.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
   
//   system("python main.py");

   PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;

   if (argc < 5)
   {
       printf("Usage: exe_name python_source function_name\n");
       return 1;
   }

   // Initialize the Python Interpreter
   Py_Initialize();

   // Build the name object
   pName = PyString_FromString(argv[1]);

   PyRun_SimpleString("import sys");
   PyRun_SimpleString("import os");
   PyRun_SimpleString("sys.path.append(os.getcwd())");

   // Load the module object
   pModule = PyImport_Import(pName);

   // pDict is a borrowed reference
   pDict = PyModule_GetDict(pModule);

   // pFunc is also a borrowed reference
   pFunc = PyDict_GetItemString(pDict, argv[2]);

   // Process extra parameters
   pArgs = PyTuple_New(argc - 3);
   for (int i=0; i < argc-3; i++) {
      pValue = PyInt_FromLong(atoi(argv[i + 3]));

      if (!pValue) {
         PyErr_Print(); return 1; }

      PyTuple_SetItem(pArgs, i, pValue);
   }

//   pValue = PyObject_CallObject(pFunc, pArgs);

   if (PyCallable_Check(pFunc)) {
      PyObject_CallObject(pFunc, pArgs);
   }
   else {
       PyErr_Print();
       Py_DECREF(pValue);
   }

   // Clean up
   Py_DECREF(pModule);
   Py_DECREF(pName);
   Py_DECREF(pArgs);

   // Finish the Python Interpreter
   Py_Finalize();

   return 0;
}
